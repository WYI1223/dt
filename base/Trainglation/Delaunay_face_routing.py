
from base.DCEL.dcel import DCEL
from base.DCEL.face import Face
from base.DCEL.halfedge import HalfEdge
from base.DCEL.vertex import Vertex
from base.DCEL.geometry import orientation,in_circle_test, midpoint
from base.GlobalTestDelunay import GlobalTestDelunay
import collections, math


class DelaunayWithRouting(DCEL):

    def __init__(self):
        super().__init__()
        self.last_locate_face = None

    def initialize(self, Scale):
        v1 = self.add_vertex(0.0 * Scale, 0.0 * Scale)
        v2 = self.add_vertex(1.0 * Scale, 0.0 * Scale)
        v3 = self.add_vertex(0.5 * Scale, math.sqrt(3) / 2 * Scale)
        initial_face = self.create_initial_triangle(v1, v2, v3)
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

    # def update_face_neighbourhood(self,face:Face):
    #     """同步 face 以及它邻接面的 neighbourhoods（保持双向、一致）。"""
    #     edges_face = self.enumerate_half_edges(face)
    #     face_neighbors = [he.twin.incident_face for he in edges_face]
    #     face.neighbourhoods = face_neighbors
    #
    #     # 反向同步
    #     for nb in face_neighbors:
    #         edges_nb = self.enumerate_half_edges(nb)
    #         nb_neighbors = []
    #         for he in edges_nb:
    #             neighbour = he.twin.incident_face
    #             # 这里才是避免 self-reference
    #             if neighbour is nb:
    #                 continue
    #             if neighbour is self.outer_face:
    #                 continue
    #             nb_neighbors.append(neighbour)
    #
    #         # 确保包含原始 face
    #         if face not in nb_neighbors:
    #             nb_neighbors.append(face)
    #
    #         nb.neighbourhoods = nb_neighbors
        # half_edges_stack = self.enumerate_half_edges(face)
        # neighbourhoods = []
        # for half_edge in half_edges_stack:
        #     neighbourhoods.append(half_edge.twin.incident_face)
        # # update
        # face.neighbourhoods = neighbourhoods
        # for neighbourhoods_face in neighbourhoods:
        #     neighbourhood_half_edges_stack = self.enumerate_half_edges(neighbourhoods_face)
        #     neighbourhoods_neighbourhoods = []
        #     for half_edge in neighbourhood_half_edges_stack:
        #         neighbour = half_edge.twin.incident_face
        #         neighbourhoods_neighbourhoods.append(neighbour)
        #     neighbourhoods_face.neighbourhoods = neighbourhoods_neighbourhoods
        #
        # print(face)
        # return neighbourhoods
    def neighbours(self, face: Face):
        """迭代返回与 `face` 相邻的全部面（去掉 None / self）"""
        for he in self.enumerate_half_edges(face):
            nb = he.twin.incident_face
            if nb is not None and nb is not self.outer_face:
                yield nb

    # def locate_face(self, x: float, y: float, face=None) -> Face:
    #     print(face)
    #     if face is None:
    #         face = self.last_locate_face
    #         # print(face)
    #
    #     face_queue = collections.deque()
    #     face_queue.append(face)
    #     r = Vertex(x,y)
    #
    #     while face_queue:
    #         face = face_queue.popleft()
    #         # print(face)
    #         # 是否在面内
    #         start = face.outer_component
    #         e = start
    #         while True:
    #             p = e.origin
    #             q = e.next.origin
    #             if orientation(p,q,r) <= 0:
    #                 break
    #             e = e.next
    #             if e == start:
    #                 return face
    #
    #         face_neigbourhoods = face.neighbourhoods
    #         if len(face_neigbourhoods) != 0:
    #             for neigbourhood in face_neigbourhoods:
    #                 if neigbourhood.visit is face:
    #                     continue
    #                 face_queue.append(neigbourhood)
    #
    #     self.draw()
    #     raise Exception("No face found, point:", r,self.faces)
    def locate_face(self, x, y, start=None):
        if start is None:
            start = self.last_locate_face

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

    from collections import deque

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
        dcel.draw()
    dcel.draw()
    print(GlobalTestDelunay(dcel))