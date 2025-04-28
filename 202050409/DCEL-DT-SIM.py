import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 工具函数
# ------------------------------

def orientation(p, q, r):
    """
    计算向量 (q - p) 与 (r - p) 的叉积；
    若结果 > 0，则 r 在有向线段 p->q 的左侧，
    若结果 < 0，则 r 在其右侧；
    若结果 = 0，则三点共线。
    计算三点 p, q, r 的方向关系，利用叉积公式：
          ToLeft(p, q, r) = | p.x p.y 1 |
                            | q.x q.y 1 | = (q.x - p.x)*(r.y - p.y) - (q.y - p.y)*(r.x - p.x)
                            | r.x r.y 1 |
        返回值 > 0 表示 r 在向量 p->q 的左侧，
               = 0 表示共线，
               < 0 表示在右侧。
    """
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

def in_circle_test(p, q, r, d):
    """
    计算 4x4 行列式（InCircle 测试）
    对于逆时针顺序的 p, q, r 来说：
      如果结果 > 0，则 d 落在由 p, q, r 构成的圆内。
    """
    M = np.array([
        [p.x, p.y, p.x**2 + p.y**2, 1],
        [q.x, q.y, q.x**2 + q.y**2, 1],
        [r.x, r.y, r.x**2 + r.y**2, 1],
        [d.x, d.y, d.x**2 + d.y**2, 1]
    ])
    return np.linalg.det(M)

# ------------------------------
# DCEL 数据结构定义
# ------------------------------

class Vertex:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.incident_edge = None  # 任一与该顶点关联的半边

    def __repr__(self):
        return f"Vertex({self.x:.2f}, {self.y:.2f})"


class HalfEdge:
    def __init__(self, origin: Vertex):
        self.origin = origin          # 起始顶点
        self.twin = None              # 对边
        self.next = None              # 同一面中下一半边
        self.prev = None              # 同一面中上一半边
        self.incident_face = None     # 所属面

        # 额外字段：例如 certificate 或其他辅助信息可在此扩展
        self.certificate = None

    def __repr__(self):
        return f"HalfEdge({self.origin})"


class Face:
    def __init__(self):
        self.outer_component = None  # 指向该面上任一半边

    def __repr__(self):
        return f"Face(outer_component={self.outer_component})"


class DCEL:
    def __init__(self):
        self.vertices = []    # 所有顶点
        self.half_edges = []  # 所有半边
        self.faces = []       # 所有面
        self.outer_face = None # 表示无限面，即外部面

    def add_vertex(self, x: float, y: float) -> Vertex:
        v = Vertex(x, y)
        self.vertices.append(v)
        return v

    def add_half_edge(self, origin: Vertex) -> HalfEdge:
        he = HalfEdge(origin)
        self.half_edges.append(he)
        return he

    def add_face(self) -> Face:
        face = Face()
        self.faces.append(face)
        return face

    def create_infinite_face(self):
        """创建并返回无限面（outer face）。"""
        # 无限面不加入 self.faces 中，而是单独记录在 outer_face
        outer = Face()
        self.outer_face = outer
        return outer

    def create_initial_triangle(self, v1: Vertex, v2: Vertex, v3: Vertex) -> Face:
        """
        利用三个顶点构造一个初始三角形（假定逆时针顺序），
        同时构造无限面（outer face），使得所有边界 twin 半边组成无限面边界环。
        """
        # 构造有限面，也就是三角形的内部面
        face = self.add_face()

        # 创建 6 个半边（每条边对应两个方向）
        he1 = self.add_half_edge(v1)
        he2 = self.add_half_edge(v2)
        he3 = self.add_half_edge(v3)
        he1_twin = self.add_half_edge(v2)
        he2_twin = self.add_half_edge(v3)
        he3_twin = self.add_half_edge(v1)

        # 设置 twin 指针
        he1.twin = he1_twin;   he1_twin.twin = he1
        he2.twin = he2_twin;   he2_twin.twin = he2
        he3.twin = he3_twin;   he3_twin.twin = he3

        # 设置有限面内的 next 和 prev 指针（逆时针顺序）
        he1.next = he2;    he2.next = he3;    he3.next = he1
        he1.prev = he3;    he2.prev = he1;    he3.prev = he2

        # 将有限面内的半边与面关联
        he1.incident_face = face
        he2.incident_face = face
        he3.incident_face = face
        face.outer_component = he1

        # 对顶点设定 incident_edge（如果尚未设置）
        for v, he in [(v1, he1), (v2, he2), (v3, he3)]:
            if v.incident_edge is None:
                v.incident_edge = he

        # 下面处理边界的 twin 半边，构造无限面
        # 若还没有无限面，则创建无限面
        if self.outer_face is None:
            outer = self.create_infinite_face()
        else:
            outer = self.outer_face

        # 我们需要构造无限面边界环，让所有未被分配（或分配为无限面）的 twin 半边组成闭合环
        # 假设有限三角形的顶点顺序为 v1, v2, v3 (逆时针)，则有限面边为:
        #   he1: v1->v2, he2: v2->v3, he3: v3->v1.
        # 它们的 twin 分别为:
        #   he1_twin: v2->v1, he2_twin: v3->v2, he3_twin: v1->v3.
        # 现在我们将把这三个 twin 半边链接成一个闭合环作为无限面边界。
        #
        # 链接规则（可以选择一种合理的顺序）：
        # 我们设定：
        #   he1_twin.next = he3_twin,    he3_twin.next = he2_twin,    he2_twin.next = he1_twin.
        # 同时设置 prev 指针：
        #   he3_twin.prev = he1_twin,    he2_twin.prev = he3_twin,    he1_twin.prev = he2_twin.
        #
        he1_twin.next = he3_twin
        he3_twin.next = he2_twin
        he2_twin.next = he1_twin

        he3_twin.prev = he1_twin
        he2_twin.prev = he3_twin
        he1_twin.prev = he2_twin

        # 设置这些 twin 半边的 incident_face 为无限面 outer
        he1_twin.incident_face = outer
        he2_twin.incident_face = outer
        he3_twin.incident_face = outer

        # 将无限面 outer 的边界指针指向任一 twin半边（例如 he1_twin）
        outer.outer_component = he1_twin

        return face
    def enumerate_vertices(self, face: Face):
        """
        遍历面 face 上的所有顶点，返回顶点列表。
        具体方法：从 face.outer_component 出发，沿着 next 指针循环遍历直到回到起点。
        """
        vertices = []
        start = face.outer_component
        e = start
        while True:
            vertices.append(e.origin)
            e = e.next
            if e == start:
                break
        return vertices

    def insert_point_in_triangle(self, face: Face, x: float, y: float):
        """
        在给定三角形 face 内插入一个新点 (x,y)。
        算法步骤：
          1. 新建顶点 P。
          2. 利用 enumerate_vertices(face) 得到原三角形的三个顶点：v1, v2, v3。
          3. 删除原三角形 face（这里简单从 self.faces 中删除）。
          4. 用新顶点将原三角形拆分为三个新三角形：
                T1: (v1, v2, P)
                T2: (v2, v3, P)
                T3: (v3, v1, P)
          5. 对共享边（与 P 有关）的半边设置 twin 指针：
                - T1 与 T2 共享边：由 v2 -> P (T1) 与 P -> v2 (T2)
                - T2 与 T3 共享边：由 v3 -> P (T2) 与 P -> v3 (T3)
                - T3 与 T1 共享边：由 P -> v1 (T1) 与 v1 -> P (T3)
        注意：此处实现为一个简化版本，主要展示如何构造新三角形并更新共享边关系；实际完整实现时，还需处理与临近三角形的关联更新。
        """
        # 1. 创建新顶点 P
        new_vertex = self.add_vertex(x, y)

        # 2. 得到原三角形的顶点列表
        verts = self.enumerate_vertices(face)
        if len(verts) != 3:
            raise Exception("当前实现仅支持在三角形内插入新点。")
        v1, v2, v3 = verts

        # 3. 删除原来面
        if face in self.faces:
            self.faces.remove(face)
        else:
            raise Exception("待拆分三角形不在 DCEL 的面列表中！")

        # 4. 分别构造三个新三角形
        # 注：这里调用 create_initial_triangle 会为每个三角形构造完整的半边，
        #       但新点 P 出现的边在不同三角形中应共享 twin 关系。
        face1 = self.create_initial_triangle(v1, v2, new_vertex)
        face2 = self.create_initial_triangle(v2, v3, new_vertex)
        face3 = self.create_initial_triangle(v3, v1, new_vertex)

        # 5. 更新共享边的 twin 指针
        # face1: vertices顺序为 (v1, v2, P)
        #   - face1 中，边从 v2 到 P：记为 he_face1_2; 边从 P 到 v1：记为 he_face1_3。
        # face2: vertices顺序为 (v2, v3, P)
        #   - face2 中，边从 P 到 v2：记为 he_face2_3.
        # face3: vertices顺序为 (v3, v1, P)
        #   - face3 中，边从 v1 到 P：记为 he_face3_2; 边从 P 到 v3：记为 he_face3_3.
        #
        # 先定位各新三角形中新点相关的半边：
        def get_edges_for_face(face, order):
            # order 为 (a, b, c) 对应 face.outer_component经过 next 得到的三个半边，
            # 返回一个列表 [e1, e2, e3]，分别为从 a->b, b->c, c->a 的半边。
            edges = []
            e = face.outer_component
            for _ in range(3):
                edges.append(e)
                e = e.next
            return edges

        edges1 = get_edges_for_face(face1, (v1, v2, new_vertex))
        edges2 = get_edges_for_face(face2, (v2, v3, new_vertex))
        edges3 = get_edges_for_face(face3, (v3, v1, new_vertex))

        # 对比各个三角形的顶点，确定与新点相关的边：
        # face1: 期望顺序： e1: v1->v2, e2: v2->new_vertex, e3: new_vertex->v1.
        # face2: 期望顺序： e1: v2->v3, e2: v3->new_vertex, e3: new_vertex->v2.
        # face3: 期望顺序： e1: v3->v1, e2: v1->new_vertex, e3: new_vertex->v3.
        he_face1 = edges1     # [v1->v2, v2->P, P->v1]
        he_face2 = edges2     # [v2->v3, v3->P, P->v2]
        he_face3 = edges3     # [v3->v1, v1->P, P->v3]

        # 共享边设置：
        # 共享边1：T1 与 T2：face1 的第二边 (v2->P) 与 face2 的第三边 (P->v2)
        he_face1[1].twin = he_face2[2]
        he_face2[2].twin = he_face1[1]

        # 共享边2：T2 与 T3：face2 的第二边 (v3->P) 与 face3 的第三边 (P->v3)
        he_face2[1].twin = he_face3[2]
        he_face3[2].twin = he_face2[1]

        # 共享边3：T3 与 T1：face3 的第二边 (v1->P) 与 face1 的第三边 (P->v1)
        he_face3[1].twin = he_face1[2]
        he_face1[2].twin = he_face3[1]

        # 返回新生成的三个面
        return [face1, face2, face3]

    def locate_face(self, x: float, y: float) -> Face:
        """
        定位包含点 (x, y) 的有限面。
        算法：从一个任意有限面出发，利用“走查法”沿邻接关系查找。
        如果走出有限面，则返回无限面。
        """
        if not self.faces:
            return None

        temp_point = Vertex(x, y)  # 临时点对象（不添加至 self.vertices）
        candidate = self.faces[0]  # 从任一有限面出发

        while True:
            verts = self.enumerate_vertices(candidate)
            inside = True
            e = candidate.outer_component
            # 遍历候选面所有边，判断点是否位于三角形内部（逆时针排列时，应对所有边满足 orientation >= 0）
            for _ in range(len(verts)):
                if orientation(e.origin, e.next.origin, temp_point) < 0:
                    # 点不在当前边左侧，转至该边对边所在的面（如果存在）
                    if e.twin and e.twin.incident_face:
                        candidate = e.twin.incident_face
                        inside = False
                        break
                    else:
                        # 如果不存在对边或有限面，则点位于无限面
                        return self.outer_face
                e = e.next
            if inside:
                return candidate

    def delete_edge(self, edge: HalfEdge):
        """
        删除指定内部边 edge（及其 twin），并将其相邻的两个有限面 f1 和 f2 合并为一个面。
        注意：不得删除构成初始超三角形的边（这里简单检查是否涉及无限面）。
        """
        # 若其中一个 incident_face为无限面，则不允许删除
        if edge.incident_face == self.outer_face or edge.twin.incident_face == self.outer_face:
            raise Exception("不能删除超三角形（无限面边界）的边。")

        f1 = edge.incident_face
        f2 = edge.twin.incident_face

        # 记相关半边：
        a = edge.prev   # f1 中 edge 之前的半边
        b = edge.next   # f1 中 edge 之后的半边
        c = edge.twin.prev   # f2 中 edge.twin 之前的半边
        d = edge.twin.next   # f2 中 edge.twin 之后的半边

        # 合并 f1 和 f2 的边界：
        # 将 f1 的边中 edge 被删除后，a 与 d 相邻；
        a.next = d
        d.prev = a
        # 将 f2 的边中 edge.twin 被删除后，c 与 b 相邻；
        c.next = b
        b.prev = c

        # 选择新合并面 f = f1，并遍历 f2 的边，将其 incident_face 设置为 f1
        current = d
        while True:
            current.incident_face = f1
            current = current.next
            if current == d:
                break

        # 设 f1.outer_component 指向 d（新周期中的一个半边）
        f1.outer_component = d

        # 从有限面列表中删除 f2
        if f2 in self.faces:
            self.faces.remove(f2)

        # 从 half_edges 列表中删除 edge 和 edge.twin
        if edge in self.half_edges:
            self.half_edges.remove(edge)
        if edge.twin in self.half_edges:
            self.half_edges.remove(edge.twin)

        return f1
    def draw(self, show=True):
        """
        使用 matplotlib 绘制当前 DCEL 中的所有面、顶点、半边及其方向箭头，
        并在每个面中心显示该面在 self.faces 中的索引。
        """
        plt.figure()
        ax = plt.gca()

        # 1. 绘制所有面（闭合多边形）并标注面索引
        for idx, face in enumerate(self.faces):
            vertices = self.enumerate_vertices(face)
            if not vertices:
                continue
            # 获取面顶点坐标，闭合多边形
            x_coords = [v.x for v in vertices] + [vertices[0].x]
            y_coords = [v.y for v in vertices] + [vertices[0].y]
            plt.plot(x_coords, y_coords, 'b-', lw=2)

            # 计算该面中心（顶点的均值，适用于三角形）
            centroid_x = sum(v.x for v in vertices) / len(vertices)
            centroid_y = sum(v.y for v in vertices) / len(vertices)
            # 绘制面索引，颜色为洋红色，居中显示
            plt.text(centroid_x, centroid_y, f"{idx}", color="magenta",
                     fontsize=12, ha="center", va="center", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # 2. 绘制所有顶点
        for v in self.vertices:
            plt.plot(v.x, v.y, 'ro')

        # 3. 绘制所有半边及其方向箭头
        for he in self.half_edges:
            # 若没有 next 则跳过（理论上所有半边都有 next）
            if he.next is None:
                continue
            # 起点与终点（he.origin 与 he.next.origin）
            x1, y1 = he.origin.x, he.origin.y
            x2, y2 = he.next.origin.x, he.next.origin.y

            # 用浅色虚线绘制边
            plt.plot([x1, x2], [y1, y2], 'c--', lw=1)

            # 计算边中点
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0

            # 计算单位方向向量
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            dxu, dyu = dx / length, dy / length

            # 箭头长度设为边长的 30%
            arrow_len = length * 0.3
            # 绘制箭头表示半边方向
            plt.arrow(mid_x, mid_y, dxu * arrow_len, dyu * arrow_len,
                      head_width=0.02, head_length=0.03, fc='g', ec='g')

        plt.axis('equal')
        plt.title("DCEL Visualization with HalfEdge Directions & Face Indices")
        if show:
            plt.show()


    def __repr__(self):
        return (f"DCEL(\n  Vertices: {self.vertices}\n  "
                f"HalfEdges: {len(self.half_edges)}\n  Faces: {self.faces}\n)")

# ------------------------------
# 示例：使用 DCEL 构造初始三角形并在其中插入新点
# ------------------------------

def main():
    # 创建 DCEL 实例
    dcel = DCEL()

    # 添加三个顶点（按逆时针顺序）构造初始三角形
    v1 = dcel.add_vertex(0.0, 0.0)
    v2 = dcel.add_vertex(1.0, 0.0)
    v3 = dcel.add_vertex(0.5, math.sqrt(3)/2)
    initial_face = dcel.create_initial_triangle(v1, v2, v3)
    print("初始 DCEL 结构：")
    print(dcel)

    # 绘制初始三角形
    print("绘制初始三角形...")
    dcel.draw()

    # 在初始三角形内插入一个新点
    new_x, new_y = 0.5, 0.3
    new_faces = dcel.insert_point_in_triangle(initial_face, new_x, new_y)
    print("插入新点后生成的面：")
    for face in new_faces:
        print("面：", face)
        print("顶点：", dcel.enumerate_vertices(face))

    # 绘制更新后的 DCEL 结构
    print("绘制更新后 DCEL 结构...")
    dcel.draw()

    dcel.insert_point_in_triangle(dcel.faces[2], 0.4, 0.4)
    dcel.draw()

    print(dcel.half_edges[7])
    dcel.delete_edge(dcel.half_edges[7])
    plt.close()
    dcel.draw()

    print(dcel.half_edges)

if __name__ == "__main__":
    main()
