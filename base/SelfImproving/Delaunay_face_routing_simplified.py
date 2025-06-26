
from base.DCEL.dcel import DCEL
from base.DCEL.face import Face
from base.DCEL.halfedge import HalfEdge
from base.DCEL.vertex import Vertex
from base.DCEL.geometry import orientation,in_circle_test, midpoint, circumcenter
from base.GlobalTestDelunay import GlobalTestDelunay
import collections, math
from typing import Dict, List, Tuple, Set



class DelaunayWithRouting(DCEL):

    def __init__(self):
        super().__init__()
        self.last_locate_face = None

    def initialize(self, Scale):
        v1 = self.add_vertex(0.0 * Scale, 0.0 * Scale)
        v2 = self.add_vertex(1.0 * Scale, 0.0 * Scale)
        v3 = self.add_vertex(0.5 * Scale, math.sqrt(3) / 2 * Scale)
        self._super_ids = {id(v1), id(v2), id(v3)}          # 记录“虚顶点”
        _ = self.create_initial_triangle(v1, v2, v3)
        self.last_locate_face = self.faces[0]


    def _sanity_check_face(self, face, tag=""):
        start = face.outer_component
        e, seen, steps = start, set(), 0
        while True:
            if e in seen:
                raise RuntimeError(f"[{tag}] duplicate edge {e}")
            seen.add(e)
            steps += 1
            if steps > 100:  # 防御
                raise RuntimeError(f"[{tag}] runaway loop >100 on face {face}")
            e = e.next
            if e is start:
                break
        # verify prev <-> next consistency
        for he in seen:
            if he.next.prev is not he or he.prev.next is not he:
                raise RuntimeError(f"[{tag}] next/prev mismatch at {he}")

    def enumerate_half_edges(self, face):
        start = face.outer_component
        e, steps = start, 0
        while True:
            yield e
            if e.next is None:
                raise RuntimeError("dangling next pointer")
            e = e.next
            steps += 1
            if e is start:
                break
            if steps > 50:  # 面最多 50 条边还没回来 → 有环裂
                raise RuntimeError(
                    f"[broken] face={id(face)}, steps={steps}, cur={e}, origin={e.origin}"
                )

    def remove_edge(self, vA: Vertex, vB: Vertex):
        heAB, heBA = self.find_halfedges(vA, vB)

        f_keep = heAB.incident_face
        f_drop = heBA.incident_face

        a_prev, a_next = heAB.prev, heAB.next  # C→A, B→C ...
        b_prev, b_next = heBA.prev, heBA.next  # D→B, A→D ...

        # -------- 1. 桥接四条指针 --------
        a_prev.next, b_next.prev = b_next, a_prev
        b_prev.next, a_next.prev = a_next, b_prev

        # -------- 2. 更新面归属 --------
        self.upgrade_incident_face(a_prev, f_keep)
        f_keep.outer_component = a_prev  # 任取一条作为入口

        # -------- 3. 清理 --------
        self.faces.remove(f_drop)
        self.half_edges.remove(heAB)
        self.half_edges.remove(heBA)

        # 修复顶点的 incident_edge（若正指向已删半边）
        if vA.incident_edge in (heAB, heBA):
            vA.incident_edge = a_next  # A→(原C)
        if vB.incident_edge in (heAB, heBA):
            vB.incident_edge = b_next  # B→(原D)

        # -------- 4. 调试护栏：确保环闭合且无重复 --------
        e = a_prev
        seen = set()
        while True:
            if e in seen:
                raise RuntimeError("remove_edge 仍有重复环！")
            seen.add(e)
            e = e.next
            if e is a_prev:
                break
        # b) remove_edge() 末尾
        self._sanity_check_face(f_keep, "remove_edge")

    def add_edge(self, vA: Vertex, vB: Vertex, face=None):
        """
        在同一面内插入对角线 vA–vB，把原多边形拆成两个面
        不论原面是三角形、四边形还是更大多边形，都安全闭环
        """

        vA = self.find_vertex(vA)
        vB = self.find_vertex(vB)
        Vs = [vA, vB]
        mid = midpoint(Vs)
        if face is None:
            face = self.locate_face(mid.x, mid.y)

        # ---------- 1. 找 heA, heB ----------
        heA = next(he for he in self.enumerate_half_edges(face) if he.origin is vA)
        heB = heA
        while heB.origin is not vB:
            heB = heB.next
            if heB is heA:
                raise ValueError("vA 和 vB 不在同一个面上")

        a_prev, a_next = heA.prev, heA  # …→A, A→…
        b_prev, b_next = heB.prev, heB  # …→B, B→…

        # ---------- 2. 创建对角线 ----------
        heAB = self.add_half_edge(vA)
        heBA = self.add_half_edge(vB)
        heAB.twin = heBA
        heBA.twin = heAB

        # ---------- 3. 8 条指针一次性桥接 ----------
        # 面 keep (原 face)
        heBA.next = a_next
        heBA.prev = b_prev
        a_next.prev = heBA
        b_prev.next = heBA

        # 面 new (face2)
        heAB.next = b_next
        heAB.prev = a_prev
        b_next.prev = heAB
        a_prev.next = heAB

        # ---------- 4. 创建 / 升级两张面 ----------
        face2 = self.add_face()
        face2.outer_component = heAB
        self.upgrade_incident_face(heAB, face2)

        face.outer_component = heBA
        self.upgrade_incident_face(heBA, face)

        # ---------- 5. 调试护栏 ----------
        self._sanity_check_face(face, "add_edge-keep")
        self._sanity_check_face(face2, "add_edge-new")

        return {face, face2}

    def isDelunayTrain(self, he: HalfEdge, p: Vertex, *, debug=False) -> bool:
        """
        Lawson-flip 证书函数：
        True  → 四点构成非德劳内，    需要翻边
        False → 合法（或在圆周上），无需翻
        """
        DELAUNAY_EPS = 1e-10
        # ---------- 1. 边界半边直接跳过 ----------
        #
        # 用 “he 或 he.twin 属于无限面” 来判断是否是外层边，
        # 不要依赖 “half_edges.index(he) < 6” 这类脆弱条件
        #
        if he.incident_face is self.outer_face or he.twin.incident_face is self.outer_face:
            if debug:
                print(f"{he} is boundary edge, skip")
            return False

        # ---------- 2. 取四个点 ----------
        A = he.twin.prev.origin  # 对角 A
        B = he.twin.origin  # he.twin 的起点 B
        C = he.twin.next.origin  # 对角 C
        D = p  # 新插入的点 D

        # ---------- 3. 空圆判定 ----------
        val = in_circle_test(C, A, B, D)  # >0: D 在圆内；<0: 圆外；=0: 圆周
        # 注意你的 in_circle_test 参数顺序

        # ---------- 4. 根据公差判断 ----------
        if val > DELAUNAY_EPS:  # 在圆内 → 违反德劳内
            if debug:
                print(f"▲ 非德劳内: △{A.index},{B.index},{C.index} 包含 D={D}, val={val}")
            return True  # 需要翻边
        else:  # 圆外或圆周
            if debug and abs(val) <= DELAUNAY_EPS:
                print(f"≈ 共圆: 视为合法, val={val}")
            return False  # 保留现边

    def neighbours(self, face: Face):
        """迭代返回与 `face` 相邻的全部面（去掉 None / self）"""
        for he in self.enumerate_half_edges(face):
            nb = he.twin.incident_face
            if nb is not None and nb is not self.outer_face:
                yield nb

    def locate_face(self, x, y, start=None):
        if start is None:
            start = self.last_locate_face
        # 保险：起点若已失效，改用 faces[0]
        if start not in self.faces:
            start = self.faces[0]

        q = collections.deque([start])
        seen = {start}
        r = Vertex(x, y)

        while q:
            f = q.popleft()
            if self.point_in_face(r, f):  # 你的 orientation 检测封装一下
                self.last_locate_face = f
                return f

            for nb in self.neighbours(f):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)

        raise RuntimeError("Point outside triangulation",x,"y:",y,start)

    def point_in_face(self,vertex: Vertex, face: Face):
        start = face.outer_component
        e = start
        while True:
            p = e.origin
            q = e.next.origin
            if orientation(p,q,vertex) <= 0:
                break
            e = e.next
            if e == start:
                return face

    def insert_point_with_certificate(self, p: Vertex):
        old_face = self.locate_face(p.x, p.y)

        # 1. 拆三角形，得到 3 张新面
        new_faces = self.insert_point_in_triangle(old_face, p.x, p.y)
        for f in new_faces:
            self._sanity_check_face(f, "insert_point")

        # 2. 初始化待检查栈 = 新面上的全部 6 条边
        edge_stack = collections.deque()
        for f in new_faces:
            edge_stack.extend(self.enumerate_half_edges(f))

        # 3. Lawson 翻边循环
        while edge_stack:
            he = edge_stack.pop()
            if he not in self.half_edges:  # 边可能已被翻掉
                continue
            if self.isDelunayTrain(he, p):
                prev_twin, next_twin = he.twin.prev, he.twin.next
                self.filp_edge(he, p)
                edge_stack.append(prev_twin)
                edge_stack.append(next_twin)
        self.last_locate_face = he.twin.prev.incident_face

    # ------------------------------------------------------------------
    #  Voronoi (finite)  ——  跳过含超顶点的三角形 & site
    # ------------------------------------------------------------------
    def build_voronoi(self, *, bbox_scale: float = 2.0) -> DCEL:
        """Return finite‑boxed Voronoi diagram **excluding super‑triangle**.
        """
        vor = DCEL()
        vor_outer = vor.create_infinite_face()

        # ---------------- 1. 圆心顶点（三角形 -> 点） ----------------
        face2v: Dict[Face, Vertex] = {}
        for f in self.faces:
            # 跳过含任一超顶点的三角形
            if any(id(he.origin) in self._super_ids for he in self.enumerate_half_edges(f)):
                continue
            a, b, c = (he.origin for he in self.enumerate_half_edges(f))
            cx, cy = circumcenter(a, b, c)
            face2v[f] = vor.add_vertex(cx, cy)

        # ---------------- 2. 包围盒 ----------------
        xs = [v.x for v in self.vertices if id(v) not in self._super_ids]
        ys = [v.y for v in self.vertices if id(v) not in self._super_ids]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w, h = x_max - x_min, y_max - y_min
        x_min -= w * (bbox_scale - 1) * 0.5
        x_max += w * (bbox_scale - 1) * 0.5
        y_min -= h * (bbox_scale - 1) * 0.5
        y_max += h * (bbox_scale - 1) * 0.5
        v_ll = vor.add_vertex(x_min, y_min)
        v_lr = vor.add_vertex(x_max, y_min)
        v_ur = vor.add_vertex(x_max, y_max)
        v_ul = vor.add_vertex(x_min, y_max)
        box_vs = [v_ll, v_lr, v_ur, v_ul]
        box_hes: List[HalfEdge] = []
        for i in range(4):
            h1 = vor.add_half_edge(box_vs[i])
            h2 = vor.add_half_edge(box_vs[(i + 1) % 4])
            h1.twin = h2; h2.twin = h1
            box_hes.append(h1)
        for i, he in enumerate(box_hes):
            he.next = box_hes[(i + 1) % 4]
            he.prev = box_hes[(i - 1) % 4]
            he.incident_face = vor_outer
        vor_outer.outer_component = box_hes[0]

        # ---------------- 3. 有限 Voronoi 边 ----------------
        edge_map: Dict[frozenset, HalfEdge] = {}
        for he in self.half_edges:
            f1, f2 = he.incident_face, he.twin.incident_face
            if self.outer_face in (f1, f2):
                continue
            if f1 not in face2v or f2 not in face2v:
                continue  # 任一侧含超顶点，跳过
            key = frozenset({f1, f2})
            if key in edge_map:
                continue
            v1, v2 = face2v[f1], face2v[f2]
            ve1 = vor.add_half_edge(v1)
            ve2 = vor.add_half_edge(v2)
            ve1.twin = ve2; ve2.twin = ve1
            edge_map[key] = ve1

        # ---------------- 4. 边界射线裁剪 ----------------
        def bbox_intersection(p: Tuple[float, float], d: Tuple[float, float]):
            px, py = p; dx, dy = d
            ts = []
            if dx: ts += [(x_min - px) / dx, (x_max - px) / dx]
            if dy: ts += [(y_min - py) / dy, (y_max - py) / dy]
            ts = [t for t in ts if t > 0]
            t = min(ts) if ts else 0.0
            return px + dx * t, py + dy * t

        for he in self.half_edges:
            # 仅处理一侧为 outer_face，另一侧不是超三角形的情况
            if (he.incident_face is self.outer_face) ^ (he.twin.incident_face is self.outer_face):
                f_in = he.twin.incident_face if he.incident_face is self.outer_face else he.incident_face
                if f_in not in face2v:
                    continue  # 对应三角形含超顶点
                vc = face2v[f_in]
                ax, ay = he.origin.x, he.origin.y
                bx, by = he.twin.origin.x, he.twin.origin.y
                dx, dy = bx - ax, by - ay
                dir_vec = (dy, -dx)  # 外法向
                ix, iy = bbox_intersection((vc.x, vc.y), dir_vec)
                v_inf = vor.add_vertex(ix, iy)
                ve = vor.add_half_edge(vc)
                ve_t = vor.add_half_edge(v_inf)
                ve.twin = ve_t; ve_t.twin = ve
                ve_t.incident_face = vor_outer
                # 暂未加入 edge_map —— 仅作用于包围盒闭合

        # ---------------- 5. 建 Voronoi 面 ----------------
        for site in self.vertices:
            if id(site) in self._super_ids:
                continue  # 跳过虚顶点
            star: List[HalfEdge] = []
            he0 = site.incident_edge
            he = he0
            while True:
                f = he.incident_face
                if f in face2v:
                    star.append((math.atan2(face2v[f].y - site.y, face2v[f].x - site.x), f))
                he = he.twin.next
                if he is he0:
                    break
            if len(star) < 2:
                continue
            star.sort()
            face_site = vor.add_face()
            ring: List[HalfEdge] = []
            for i in range(len(star)):
                f_cur = star[i][1]
                f_nxt = star[(i + 1) % len(star)][1]
                key = frozenset({f_cur, f_nxt})
                ve = edge_map.get(key)
                if ve is None:
                    continue
                if ve.origin is not face2v[f_cur]:
                    ve = ve.twin
                ring.append(ve)
            n = len(ring)
            for i, ve in enumerate(ring):
                ve.incident_face = face_site
                ve.next = ring[(i + 1) % n]
                ve.prev = ring[(i - 1) % n]
            if ring:
                face_site.outer_component = ring[0]

        # ---------------- 6. 修补包围盒射线指针 ----------------
        # 将所有属于 outer_face 的半边按极角排序缝合成环；
        # 若 incident_face 仍为空，也归到 outer_face。
        cx, cy = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
        outer_edges: List[HalfEdge] = []
        for he in vor.half_edges:
            if he.incident_face is None:
                he.incident_face = vor_outer
            if he.incident_face is vor_outer:
                outer_edges.append(he)
        # 去重：只保留 origin 在 bbox 上的那一向量（剔除 twin）
        outer_edges = [he for he in outer_edges if he.origin.x in (x_min, x_max) or he.origin.y in (y_min, y_max)]
        outer_edges.sort(key=lambda h: math.atan2(h.origin.y - cy, h.origin.x - cx))
        m = len(outer_edges)
        for i, he in enumerate(outer_edges):
            he.next = outer_edges[(i + 1) % m]
            he.prev = outer_edges[(i - 1) % m]
        vor_outer.outer_component = outer_edges[0] if outer_edges else vor_outer.outer_component

        return vor
    # ======================== helpers ===========================
    @staticmethod
    def _circumcenter_coords(face: "Face") -> Tuple[float, float]:
        a = face.outer_component.origin
        b = face.outer_component.next.origin
        c = face.outer_component.next.next.origin
        d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
        ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y) + (b.x * b.x + b.y * b.y) * (c.y - a.y) +
              (c.x * c.x + c.y * c.y) * (a.y - b.y)) / d
        uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x) + (b.x * b.x + b.y * b.y) * (a.x - c.x) +
              (c.x * c.x + c.y * c.y) * (b.x - a.x)) / d
        return ux, uy

    @staticmethod
    def _add_to_ring(rings: Dict[int, "HalfEdge"], vid: int, h: "HalfEdge"):
        """把半边 h 按 vertex-id 尾插入口袋环，保持双向链闭合。"""
        if vid not in rings:
            h.next = h.prev = h
            rings[vid] = h
        else:
            fst = rings[vid];
            lst = fst.prev
            lst.next = h;
            h.prev = lst
            h.next = fst;
            fst.prev = h
    @staticmethod
    def _close_outer_ring(vor: "DCEL", outer: "Face"):
        es = [he for he in vor.half_edges if he.incident_face is outer]
        if not es: return
        cx = sum(e.origin.x for e in es) / len(es)
        cy = sum(e.origin.y for e in es) / len(es)
        es.sort(key=lambda e: math.atan2(e.origin.y - cy, e.origin.x - cx))
        for i, e in enumerate(es):
            e.next = es[(i + 1) % len(es)]
            e.prev = es[(i - 1) % len(es)]
        outer.outer_component = es[0]

def compute_delaunay(vertexs,draw=False):
    # 创建 DCEL 实例
    dcel = DelaunayWithRouting()
    Scale = 20
    dcel.initialize(20)
    for i in vertexs:
        dcel.insert_point_with_certificate(i)
        if draw:
            dcel.draw_science()
    return dcel


if __name__ == '__main__':
    vertexs = [Vertex(0.5, 0.3),
               Vertex(0.3, 0.4),
               Vertex(0.4, 0.1),
               Vertex(0.6, 0.4),
               Vertex(0.3,0.2),
               Vertex(0.5,0.45),
               Vertex(0.6,0.2),
               Vertex(0.7,0.35),
               Vertex(0.7,0.1),]
    dcel = DelaunayWithRouting()
    sacle = 1
    num = len(vertexs)
    dcel.initialize(sacle)
    for i in range(num):
        dcel.insert_point_with_certificate(vertexs[i])
        # for f in dcel.faces:
        #     dcel._sanity_check_face(f,"insert_point")
        # dcel.draw()
    dcel.draw()
    import matplotlib.pyplot as plt

    vor_box = dcel.build_voronoi()  # 有限版
    # vor_inf = dcel.build_voronoi(keep_infinite=True)

    DCEL.draw_dcel(vor_box, title="Voronoi boxed")
    # DCEL.draw_dcel(vor_inf, title="Voronoi w/ rays")
    plt.show()
    # vor_inf.draw()
    vor_box.draw()

    print(GlobalTestDelunay(dcel))