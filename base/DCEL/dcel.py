import math
import numpy as np
import matplotlib.pyplot as plt

from base.DCEL.vertex import Vertex
from base.DCEL.halfedge import HalfEdge
from base.DCEL.face import Face
from base.DCEL.geometry import orientation, in_circle_test, midpoint

class DCEL:
    def __init__(self):
        self.vertices = []    # 所有顶点
        self.half_edges = []  # 所有半边
        self.faces = []       # 所有面（有限面）
        self.outer_face = None  # 无限面

    def add_vertex(self, x: float, y: float) -> Vertex:
        v = Vertex(x, y)
        self.vertices.append(v)
        return v

    def add_half_edge(self, origin: Vertex) -> HalfEdge:
        he = HalfEdge(origin)
        self.half_edges.append(he)
        origin.incident_edge = he
        return he

    def add_face(self) -> Face:
        face = Face()
        self.faces.append(face)
        return face

    def create_infinite_face(self):
        """创建并返回无限面（outer face），不加入 self.faces 中，而单独记录。"""
        outer = Face()
        self.outer_face = outer
        return outer

    def create_initial_triangle(self, v1: Vertex, v2: Vertex, v3: Vertex) -> Face:
        """
        利用三个顶点构造初始三角形（假定逆时针顺序），同时构造无限面，
        将所有边界 twin 半边组织成无限面边界环。
        """
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

        # 设置有限面内的 next 与 prev 指针（逆时针顺序）
        he1.next = he2;    he2.next = he3;    he3.next = he1
        he1.prev = he3;    he2.prev = he1;    he3.prev = he2

        # 将有限面内的半边与面关联
        he1.incident_face = face
        he2.incident_face = face
        he3.incident_face = face
        face.outer_component = he1

        # 为顶点设置 incident_edge（若未设置）
        for v, he in [(v1, he1), (v2, he2), (v3, he3)]:
            if v.incident_edge is None:
                v.incident_edge = he

        # 处理边界 twin 半边，构造无限面
        if self.outer_face is None:
            outer = self.create_infinite_face()
        else:
            outer = self.outer_face

        # 链接这三个 twin 半边形成闭合环
        he1_twin.next = he3_twin
        he3_twin.next = he2_twin
        he2_twin.next = he1_twin

        he3_twin.prev = he1_twin
        he2_twin.prev = he3_twin
        he1_twin.prev = he2_twin

        # 设置 twin 半边的 incident_face 为无限面
        he1_twin.incident_face = outer
        he2_twin.incident_face = outer
        he3_twin.incident_face = outer

        # 无限面边界指向任意一个 twin 半边
        outer.outer_component = he1_twin

        return face

    def enumerate_vertices(self, face: Face):
        """
        从 face.outer_component 开始沿 next 遍历所有顶点，返回顶点列表。
        """
        vertices = []
        start = face.outer_component
        e = start
        count = 0
        while True:
            vertices.append(e.origin)
            e = e.next
            count += 1
            if count == 20:
                print(face,self.faces.index(face),"----------------------------")
                break
            if e == start:
                break
        return vertices

    def enumerate_half_edges(self, face: Face):
        """
        从 face.outer_component 出发遍历该面内所有半边，返回列表。
        """
        half_edges = []
        start = face.outer_component
        e = start
        while True:
            half_edges.append(e)
            e = e.next
            if e == start:
                break
        return half_edges

    def upgrade_incident_face(self, he: HalfEdge, face: Face):
        """
        从某半边出发，沿 next 更新同一循环中所有半边的 incident_face。
        """
        e = he
        while True:
            e.incident_face = face
            e = e.next
            if e == he:
                break
        return face

    def insert_point_in_triangle(self, face: Face, x: float, y: float):
        """
        在给定三角形 face 内插入新点 (x,y)，将原三角形拆分为三个新三角形。
        注意：此实现为简化版本，主要展示如何构造新三角形及更新边关系。
        """
        new_vertex = self.add_vertex(x, y)
        faceA = self.add_face()
        faceB = self.add_face()
        faceC = self.add_face()

        verts = self.enumerate_vertices(face)
        if len(verts) != 3:
            raise Exception("仅支持在三角形内插入新点。")
        v1, v2, v3 = verts

        # 删除原三角形面
        if face in self.faces:
            self.faces.remove(face)
        else:
            raise Exception("待拆分三角形不在面列表中！")

        # 创建新半边
        hePA = self.add_half_edge(new_vertex)
        hePB = self.add_half_edge(new_vertex)
        hePC = self.add_half_edge(new_vertex)
        heAP = self.add_half_edge(v1)
        heBP = self.add_half_edge(v2)
        heCP = self.add_half_edge(v3)

        # 设置每个新面对应的半边
        faceA.outer_component = hePA
        faceB.outer_component = hePB
        faceC.outer_component = hePC

        hePA.twin = heAP;   heAP.twin = hePA
        hePB.twin = heBP;   heBP.twin = hePB
        hePC.twin = heCP;   heCP.twin = hePC

        hePA.prev = heBP
        hePB.prev = heCP
        hePC.prev = heAP

        # 寻找原三角形中对应的半边
        old_hes = self.enumerate_half_edges(face)
        for he in old_hes:
            if he.origin == v1:
                heAB = he
            if he.origin == v2:
                heBC = he
            if he.origin == v3:
                heCA = he

        heAB.next = heBP
        heAB.prev = hePA
        hePA.next = heAB
        heBP.next = hePA
        heBP.prev = heAB
        faceA = self.upgrade_incident_face(heAB, faceA)

        heBC.next = heCP
        heBC.prev = hePB
        hePB.next = heBC
        heCP.next = hePB
        heCP.prev = heBC
        faceB = self.upgrade_incident_face(heBC, faceB)

        heCA.next = heAP
        heCA.prev = hePC
        hePC.next = heCA
        heAP.next = hePC
        heAP.prev = heCA
        faceC = self.upgrade_incident_face(heCA, faceC)

        return [faceA, faceB, faceC]

    def locate_face(self, x: float, y: float) -> Face:
        """
        定位包含点 (x, y) 的有限面。若点不在任何有限面内，则返回无限面。
        """
        if not self.faces:
            return None
        r = Vertex(x, y)
        for face in self.faces:
            start = face.outer_component
            e = start
            while True:
                p = e.origin
                q = e.next.origin
                if orientation(p,q,r) <= 0:
                    break
                e = e.next
                if e == start:
                    return face
        raise Exception("No face found, point:", r,self.faces)

        # if not self.faces:
        #     return None
        #
        # temp_point = Vertex(x, y)
        # candidate = self.faces[0]
        # while True:
        #     verts = self.enumerate_vertices(candidate)
        #     inside = True
        #     e = candidate.outer_component
        #     for _ in range(len(verts)):
        #         if orientation(e.origin, e.next.origin, temp_point) < 0:
        #             if e.twin and e.twin.incident_face:
        #                 candidate = e.twin.incident_face
        #                 inside = False
        #                 break
        #             else:
        #                 return self.outer_face
        #         e = e.next
        #     if inside:
        #         return candidate

    def find_vertex(self, point1: Vertex) -> Vertex:
        for _ in self.vertices:
            if _ == point1:
                return _
        print("!!!Can not find Vertex", point1)

    def find_halfedges(self, point1: Vertex, point2: Vertex):
        point1 = self.find_vertex(point1)
        point2 = self.find_vertex(point2)
        halfEdges = []
        e = point1.incident_edge
        while True:
            print(e)
            if e.twin.origin == point2:
                halfEdges.append(e)
                halfEdges.append(e.twin)
                break
            e = e.twin.next
            if e == point1.incident_edge:
                print("找不到另一个点")
        return halfEdges

    def remove_edge(self, point1: Vertex, point2: Vertex):
        """
              ------------>    ----------->
         C    <------------ B  <-----------  F
                            ^|
                            ||
                            ||
                            |v
         D  --------------> A  ----------->  E
            <-------------    <------------
        :param point1:
        :param point2:
        :return:
        """
        heAB, heBA = self.find_halfedges(point1,point2)
        heBC = heAB.next
        heDA = heAB.prev
        heFB = heBA.prev
        heAE = heBA.next

        heBC.prev = heFB
        heFB.next = heBC
        heDA.next = heAE
        heAE.prev = heDA

        self.half_edges.remove(heBA)
        self.half_edges.remove(heAB)
        point1.incident_edge = heAE
        point2.incident_edge = heBC
        self.faces.remove(heFB.incident_face)
        self.upgrade_incident_face(heDA,heDA.incident_face)
        heDA.incident_face.outer_component = heDA
        return

    def add_edge(self, point1: Vertex, point2: Vertex, face = None):
        """
              ------------>    ----------->
         C    <------------ B  <-----------  F
                            ^|
                            ||
                            ||
                            |v
         D  --------------> A  ----------->  E
            <-------------    <------------

        :param point1:
        :param point2:
        :return:
        """
        point1 = self.find_vertex(point1)
        point2 = self.find_vertex(point2)

        Vs = [point1, point2]
        mid = midpoint(Vs)
        if face is None:
            face = self.locate_face(mid.x, mid.y)
        edges = []
        vertexs = []
        half_edges = self.enumerate_half_edges(face)
        for e in half_edges:
            if e.origin == point1:
                edges.append(e)
                vertexs.append(e.origin)
            if e.origin == point2:
                edges.append(e)
                vertexs.append(e.origin)
            if len(edges) == 2:
                break
        if vertexs[0] is point2:  # 方向反了
            edges.reverse()
            vertexs.reverse()

        heAB = self.add_half_edge(vertexs[0])
        heBA = self.add_half_edge(vertexs[1])


        heAE = edges[0]
        heBC = edges[1]
        heFB = heBC.prev
        heDA = heAE.prev

        heAB.twin = heBA
        heBA.twin = heAB

        heAB.next = heBC
        heAB.prev = heDA
        heBC.prev = heAB
        heDA.next = heAB

        heBA.next = heAE
        heBA.prev = heFB
        heAE.prev = heBA
        heFB.next = heBA

        heBA.incident_face = face
        face.outer_component = heBA
        face2 = self.add_face()
        face2.outer_component = heAB
        heAB.incident_face = face2
        face2 = self.upgrade_incident_face(heAB, face2)
        return

    def filp_edge(self, he: HalfEdge, point:Vertex):
        hePointA = he.origin
        hePointB = he.next.origin
        oppositePoint = he.twin.prev.origin
        self.remove_edge(hePointA, hePointB)
        self.add_edge(oppositePoint, point)
        print("flip edge:",hePointA,hePointB,"instead by",oppositePoint,point)
        return
    def isDelunayTrain(self,half_edge: HalfEdge,point: Vertex):

        if self.half_edges.index(half_edge) < 6:
            print(half_edge,"is outside edge, skip")
            return False


        pointA = half_edge.twin.prev.origin
        pointB = half_edge.twin.origin
        pointC = half_edge.twin.next.origin

        # 两侧测试

        # 空园测试
        certifi = in_circle_test(pointC,pointA,pointB,point)

        if certifi < 0:
            print("Tran:",self.vertices.index(pointA),"",
                  self.vertices.index(pointB),"",
                  self.vertices.index(pointC),"are qualified with Point x:",
                    point.x,"y:",point.y,certifi
                  )
            return False
        elif certifi > 0:
            print("Tran:", self.vertices.index(pointA), "",
                  self.vertices.index(pointB), "",
                  self.vertices.index(pointC), "are including Point x:",
                  point.x, "y:", point.y, certifi
                  )
            return True
        else:
            raise Exception("4点共圆！！！")

    def insert_point_with_certificate(self, point: Vertex):

        face = self.locate_face(point.x, point.y)
        half_edges = self.enumerate_half_edges(face)

        for he in half_edges:
            he.certificate = self.isDelunayTrain(he,point)

        self.insert_point_in_triangle(face,point.x,point.y)

        for he in half_edges:
            if he.certificate:
                self.filp_edge(he,point)
                # hePointA = he.origin
                # hePointB = he.next.origin
                # oppositePoint = he.twin.prev.origin
                # self.remove_edge(hePointA,hePointB)
                # self.add_edge(oppositePoint,point)
        return

    def draw_old(self, show=True):
        """
        利用 matplotlib 绘制当前 DCEL 的所有面、顶点及半边（带箭头表示方向）。
        """
        plt.figure()
        ax = plt.gca()

        # 绘制所有有限面（闭合多边形）并标注面索引
        for idx, face in enumerate(self.faces):
            vertices = self.enumerate_vertices(face)
            if not vertices:
                continue
            x_coords = [v.x for v in vertices] + [vertices[0].x]
            y_coords = [v.y for v in vertices] + [vertices[0].y]
            plt.plot(x_coords, y_coords, 'b-', lw=2)
            centroid_x = sum(v.x for v in vertices) / len(vertices)
            centroid_y = sum(v.y for v in vertices) / len(vertices)
            plt.text(centroid_x, centroid_y, f"{idx}", color="magenta",
                     fontsize=12, ha="center", va="center", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # 绘制所有顶点
        for v in self.vertices:
            plt.plot(v.x, v.y, 'ro')

        # 绘制所有半边及其箭头
        for he in self.half_edges:
            if he.next is None:
                continue
            x1, y1 = he.origin.x, he.origin.y
            x2, y2 = he.next.origin.x, he.next.origin.y
            plt.plot([x1, x2], [y1, y2], 'c--', lw=1)
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            dxu, dyu = dx / length, dy / length
            arrow_len = length * 0.3
            plt.arrow(mid_x, mid_y, dxu * arrow_len, dyu * arrow_len,
                      head_width=0.02, head_length=0.03, fc='g', ec='g')

        plt.axis('equal')
        plt.title("DCEL Visualization with HalfEdge Directions & Face Indices")
        if show:
            plt.show()

    def draw_edge_index(self, show=True):
        """
        利用 matplotlib 绘制当前 DCEL 的所有面、顶点及半边（带箭头表示方向）。
        同时在每个半边旁标注其在 self.half_edges 中的索引。
        """
        plt.figure()
        ax = plt.gca()

        # 绘制所有有限面（闭合多边形）并标注面索引
        for idx, face in enumerate(self.faces):
            vertices = self.enumerate_vertices(face)
            if not vertices:
                continue
            x_coords = [v.x for v in vertices] + [vertices[0].x]
            y_coords = [v.y for v in vertices] + [vertices[0].y]
            plt.plot(x_coords, y_coords, 'b-', lw=2)
            centroid_x = sum(v.x for v in vertices) / len(vertices)
            centroid_y = sum(v.y for v in vertices) / len(vertices)
            plt.text(centroid_x, centroid_y, f"{idx}", color="magenta",
                     fontsize=12, ha="center", va="center",
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # 绘制所有顶点
        for v in self.vertices:
            plt.plot(v.x, v.y, 'ro')

        # 绘制所有半边及其箭头，并标注每个半边的 index
        for idx, he in enumerate(self.half_edges):
            if he.next is None:
                continue
            x1, y1 = he.origin.x, he.origin.y
            x2, y2 = he.next.origin.x, he.next.origin.y
            plt.plot([x1, x2], [y1, y2], 'c--', lw=1)

            # 计算半边的中点
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0

            # 计算单位方向向量
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            dxu, dyu = dx / length, dy / length

            # 计算法向量用于偏移标注位置（这里采用 (-dyu, dxu) 作为单位法向量）
            offset = 0.05 * length  # 偏移量可以根据需要调整
            text_x = mid_x - dyu * offset
            text_y = mid_y + dxu * offset

            # 在半边附近标注其 index
            plt.text(text_x, text_y, f"{idx}", color="black",
                     fontsize=10, ha="center", va="center",
                     bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))

            # 绘制箭头指示方向
            arrow_len = length * 0.3
            plt.arrow(mid_x, mid_y, dxu * arrow_len, dyu * arrow_len,
                      head_width=0.02, head_length=0.03, fc='g', ec='g')

        plt.axis('equal')
        plt.title("DCEL Visualization with HalfEdge Directions, Face & HalfEdge Indices")
        if show:
            plt.show()

    def draw(self, show=True):
        plt.figure()
        ax = plt.gca()

        # 绘制所有有限面（闭合多边形）并标注面索引
        for idx, face in enumerate(self.faces):
            vertices = self.enumerate_vertices(face)
            if not vertices:
                continue
            x_coords = [v.x for v in vertices] + [vertices[0].x]
            y_coords = [v.y for v in vertices] + [vertices[0].y]
            plt.plot(x_coords, y_coords, 'b-', lw=2)
            centroid_x = sum(v.x for v in vertices) / len(vertices)
            centroid_y = sum(v.y for v in vertices) / len(vertices)
            plt.text(centroid_x, centroid_y, f"{idx}", color="magenta",
                     fontsize=12, ha="center", va="center",
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # 绘制所有顶点，并在旁边标注顶点在 self.vertices 中的索引
        for idx, v in enumerate(self.vertices):
            plt.plot(v.x, v.y, 'ro')
            # 调整偏移量避免文字与顶点重合，可以根据实际情况修改偏移值
            plt.text(v.x + 0.02, v.y + 0.02, f"{idx}", color="blue", fontsize=10)

        # 绘制所有半边及其箭头
        for he in self.half_edges:
            if he.next is None:
                continue
            x1, y1 = he.origin.x, he.origin.y
            x2, y2 = he.next.origin.x, he.next.origin.y
            plt.plot([x1, x2], [y1, y2], 'c--', lw=1)
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            dxu, dyu = dx / length, dy / length
            arrow_len = length * 0.3
            plt.arrow(mid_x, mid_y, dxu * arrow_len, dyu * arrow_len,
                      head_width=0.02, head_length=0.03, fc='g', ec='g')

        plt.axis('equal')
        plt.title("DCEL Visualization with HalfEdge Directions, Face & Vertex Indices")
        if show:
            plt.show()

    def draw_science(self, show=True,
             draw_vertices=True,
             draw_halfedges=True,
             draw_faces=True,
             only_hull=False):
        plt.figure()
        ax = plt.gca()

        # 定义调色板：使用 matplotlib 的 tab10
        cmap = plt.cm.get_cmap('tab10')
        face_color = cmap(2)
        vertex_color = cmap(0)
        arrow_color = cmap(1)

        # 辅助函数：获取顶点在 self.vertices 中的索引
        def get_index(v):
            try:
                return self.vertices.index(v)
            except ValueError:
                return None

        initial_indices = {0, 1, 2}

        # 1) 绘制所有有限面（闭合多边形）边段，位于最底层 zorder=1
        if draw_faces:
            for face in self.faces:
                verts = self.enumerate_vertices(face)
                if not verts:
                    continue
                n = len(verts)
                for i in range(n):
                    v1 = verts[i]
                    v2 = verts[(i + 1) % n]
                    idx1 = get_index(v1)
                    idx2 = get_index(v2)
                    # 如果仅绘制凸包，跳过与初始顶点相连的边段
                    if only_hull and (idx1 in initial_indices or idx2 in initial_indices):
                        continue
                    ax.plot(
                        [v1.x, v2.x], [v1.y, v2.y],
                        color=face_color,
                        linewidth=2,
                        zorder=1
                    )

        # 2) 绘制所有顶点，位于中间 zorder=2
        if draw_vertices:
            for v in self.vertices:
                ax.plot(
                    v.x, v.y,
                    marker='o',
                    color=vertex_color,
                    zorder=2
                )

        # 3) 绘制所有半边方向箭头（仅方向），位于最顶层 zorder=3
        if draw_halfedges:
            for he in self.half_edges:
                if he.next is None:
                    continue
                # 不再根据 only_hull 过滤半边
                x1, y1 = he.origin.x, he.origin.y
                x2, y2 = he.next.origin.x, he.next.origin.y
                mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length == 0:
                    continue
                dxu, dyu = dx / length, dy / length
                arrow_len = length * 0.3
                ax.arrow(
                    mid_x, mid_y,
                    dxu * arrow_len, dyu * arrow_len,
                    head_width=0.02,
                    head_length=0.03,
                    fc=arrow_color,
                    ec=arrow_color,
                    zorder=3
                )

        # 设置坐标轴与标题
        ax.axis('equal')
        ax.set_title("DCEL Visualization")
        from matplotlib.lines import Line2D
        # 构造图例
        legend_handles = []
        if draw_faces:
            legend_handles.append(
                Line2D([0], [0], color=face_color, linewidth=2, label='Face boundary')
            )
        if draw_vertices:
            legend_handles.append(
                Line2D([0], [0], marker='o', color=vertex_color, linestyle='None', label='Vertex')
            )
        if draw_halfedges:
            legend_handles.append(
                Line2D([0], [0], color=arrow_color, marker=r'$\rightarrow$', markersize=10, linestyle='None',
                       label='Half-edge direction')
            )
        if legend_handles:
            ax.legend(handles=legend_handles, loc='best')

        if show:
            plt.show()

    def __repr__(self):
        return (f"DCEL(\n  Vertices: {self.vertices}\n  "
                f"HalfEdges: {len(self.half_edges)}\n  Faces: {self.faces}\n)")
