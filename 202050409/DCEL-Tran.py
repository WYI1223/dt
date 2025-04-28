import math
import matplotlib.pyplot as plt

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

    def create_initial_triangle(self, v1: Vertex, v2: Vertex, v3: Vertex) -> Face:
        """
        利用三个顶点构造一个初始三角形（假定逆时针顺序），构造对应的 DCEL 结构。
        """
        face = self.add_face()

        # 创建 6 个半边（每条边两个方向）
        he1 = self.add_half_edge(v1)
        he2 = self.add_half_edge(v2)
        he3 = self.add_half_edge(v3)
        he1_twin = self.add_half_edge(v2)
        he2_twin = self.add_half_edge(v3)
        he3_twin = self.add_half_edge(v1)

        # 设置 twin 指针
        he1.twin = he1_twin
        he1_twin.twin = he1
        he2.twin = he2_twin
        he2_twin.twin = he2
        he3.twin = he3_twin
        he3_twin.twin = he3

        # 设置 next 和 prev 指针（面内循环，逆时针顺序）
        he1.next = he2;    he2.next = he3;    he3.next = he1
        he1.prev = he3;    he2.prev = he1;    he3.prev = he2

        # 将半边与面关联（这里只关联内部面，不处理无限面）
        he1.incident_face = face
        he2.incident_face = face
        he3.incident_face = face

        face.outer_component = he1

        # 对顶点设定 incident_edge（若未设置）
        for v, he in [(v1, he1), (v2, he2), (v3, he3)]:
            if v.incident_edge is None:
                v.incident_edge = he

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

    # ------------------------------
    # 以下为新辅助函数：从给定顶点构造一个新三角形面
    # 该函数将创建新的半边（不重用任何已有半边），并返回新面及其半边列表（逆时针顺序）
    # ------------------------------
    def create_triangle_face(self, v1: Vertex, v2: Vertex, v3: Vertex):
        """
        根据给定顶点 v1, v2, v3 (按逆时针顺序) 构造一个新面。
        返回 (face, [e1, e2, e3])，其中 e1: v1->v2, e2: v2->v3, e3: v3->v1。
        """
        face = self.add_face()

        # 创建3条半边（面内的边）
        e1 = self.add_half_edge(v1)
        e2 = self.add_half_edge(v2)
        e3 = self.add_half_edge(v3)

        # 创建对应的 twin 半边，为后续可能建立内部边共享链接预留（初始时置为 None ）
        e1_twin = self.add_half_edge(v2)
        e2_twin = self.add_half_edge(v3)
        e3_twin = self.add_half_edge(v1)

        # 建立 twin 关系（默认各自配对，后续如果与其他面共享则修正）
        e1.twin = e1_twin;  e1_twin.twin = e1
        e2.twin = e2_twin;  e2_twin.twin = e2
        e3.twin = e3_twin;  e3_twin.twin = e3

        # 设置环（内部边界，逆时针顺序）
        e1.next = e2;  e2.next = e3;  e3.next = e1
        e1.prev = e3;  e2.prev = e1;  e3.prev = e2

        # 设置面关联
        e1.incident_face = face
        e2.incident_face = face
        e3.incident_face = face
        face.outer_component = e1

        # 更新顶点 incident_edge（如果为空）
        for v, he in [(v1, e1), (v2, e2), (v3, e3)]:
            if v.incident_edge is None:
                v.incident_edge = he

        return face, [e1, e2, e3]

    # ------------------------------
    # 改进后的插入操作：在一个三角形内插入新点，不复用旧半边
    # ------------------------------
    def insert_point_in_triangle_improved(self, face: Face, x: float, y: float):
        """
        在给定三角形 face 内插入新点 (x, y)。
        做法：
          1. 利用 enumerate_vertices 得到原三角形顶点 [v1, v2, v3]。
          2. 创建新顶点 P。
          3. 删除面 face 以及它的所有边（从 self.faces 和 self.half_edges 中移除）。
          4. 利用原面外边界（v1->v2, v2->v3, v3->v1）重新构造三新面时，将这三条边重新分配：
             - 新外边保持不变，分别分配给各新面；
             - 创建三条新内部边（互为 twin）连接新顶点与原三角形各顶点。
          5. 构造三个新三角形：
               T1: [v1, v2, P]
               T2: [v2, v3, P]
               T3: [v3, v1, P]
        """
        # 1. 得到原面顶点
        old_vertices = self.enumerate_vertices(face)
        if len(old_vertices) != 3:
            raise Exception("当前实现仅支持三角形内部点插入。")
        v1, v2, v3 = old_vertices

        # 2. 创建新顶点 P
        new_vertex = self.add_vertex(x, y)

        # 3. 删除原面及其边
        # 找到原面所有边（假设三角形，因此循环3次）
        old_edges = []
        e = face.outer_component
        for _ in range(3):
            old_edges.append(e)
            e = e.next
        # 从 DCEL 中移除这些边
        for he in old_edges:
            if he in self.half_edges:
                self.half_edges.remove(he)
            # 不删除 twin 边，因为它们可能属于无限面（外部），但这里不再使用它们
        # 从面列表中删除旧面
        if face in self.faces:
            self.faces.remove(face)

        # 4. 新三角形的外边直接沿用原面边，但因删除原边，
        #    我们假设原外边可以重新创建；实际上这里我们采用不复用原边的方案，
        #    即外边也重新创建（注意：原外边的 twin 半边在无限面中需要保持不变，
        #    但此处简化处理，我们只处理有限面）。
        # 构造新三角形时，外边由原顶点间边直接新建：
        # T1: (v1, v2, P), T2: (v2, v3, P), T3: (v3, v1, P)

        # 调用新辅助函数分别创建三个三角形面：
        face1, edges1 = self.create_triangle_face(v1, v2, new_vertex)  # T1: 边： e1: v1->v2, e2: v2->P, e3: P->v1
        face2, edges2 = self.create_triangle_face(v2, v3, new_vertex)  # T2: 边： f1: v2->v3, f2: v3->P, f3: P->v2
        face3, edges3 = self.create_triangle_face(v3, v1, new_vertex)  # T3: 边： g1: v3->v1, g2: v1->P, g3: P->v3

        # 5. 共享内部边建立：三个新三角形共享的新边为
        #    （a）T1 与 T2 共享边：v2 <-> P => T1 中边 edges1[1] (v2->P) 与 T2 中边 edges2[2] (P->v2)
        #    （b）T2 与 T3 共享边：v3 <-> P => T2 中边 edges2[1] (v3->P) 与 T3 中边 edges3[2] (P->v3)
        #    （c）T3 与 T1 共享边：v1 <-> P => T3 中边 edges3[1] (v1->P) 与 T1 中边 edges1[2] (P->v1)
        edges1[1].twin = edges2[2]
        edges2[2].twin = edges1[1]

        edges2[1].twin = edges3[2]
        edges3[2].twin = edges2[1]

        edges3[1].twin = edges1[2]
        edges1[2].twin = edges3[1]

        # 外部边不进行共享（它们各自属于不同新面），例如 T1 中 edges1[0] 是 v1->v2，
        # T2 中 edges2[0] 是 v2->v3，T3 中 edges3[0] 是 v3->v1

        # 将新面加入 DCEL 的面列表
        self.faces.extend([face1, face2, face3])

        return [face1, face2, face3]

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
    new_faces = dcel.insert_point_in_triangle_improved(initial_face, new_x, new_y)
    print("插入新点后生成的面：")
    print(dcel)


    # 绘制更新后的 DCEL 结构
    print("绘制更新后 DCEL 结构...")
    dcel.draw()

    dcel.insert_point_in_triangle_improved(dcel.faces[2], 0.4, 0.4)
    dcel.draw()
    print(dcel)


    dcel.insert_point_in_triangle_improved(dcel.faces[0], 0.3, 0.1)
    dcel.draw()
    print(dcel)

    print(repr(dcel.faces[2]))

if __name__ == "__main__":
    main()
