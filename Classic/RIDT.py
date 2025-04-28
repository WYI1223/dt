import random
import math
import matplotlib.pyplot as plt


class Vertex:
    """ DCEL中的顶点 """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.incident_edge = None

    def __repr__(self):
        return f"V({self.x:.2f},{self.y:.2f})"


class HalfEdge:
    def __init__(self):
        self.origin = None
        self.twin = None
        self.next = None
        self.prev = None
        self.face = None

    def __repr__(self):
        return f"HE({self.origin})"

    def endpoints(self):
        return self.origin, self.next.origin


class Face:
    def __init__(self):
        self.edge = None

    def __repr__(self):
        vs = self.vertices()
        if len(vs) == 3:
            return f"F({vs[0]}, {vs[1]}, {vs[2]})"
        else:
            return "F(?)"

    def vertices(self):
        vs = []
        e = self.edge
        if e is None:
            return vs
        start = e
        vs.append(e.origin)
        e = e.next
        while e != start:
            vs.append(e.origin)
            e = e.next
        return vs


class DCEL:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.faces = []

    def create_face(self, v1, v2, v3):
        e1, e2, e3 = HalfEdge(), HalfEdge(), HalfEdge()
        self.edges.extend([e1, e2, e3])

        f = Face()
        self.faces.append(f)

        e1.origin = v1
        e2.origin = v2
        e3.origin = v3

        e1.next = e2
        e2.next = e3
        e3.next = e1

        e1.prev = e3
        e2.prev = e1
        e3.prev = e2

        e1.face = f
        e2.face = f
        e3.face = f

        f.edge = e1

        v1.incident_edge = e1
        v2.incident_edge = e2
        v3.incident_edge = e3

        return f


def orientation(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def in_circle(a, b, c, d):
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y
    dx, dy = d.x, d.y

    mat = [
        [ax - dx, ay - dy, (ax * ax + ay * ay) - (dx * dx + dy * dy)],
        [bx - dx, by - dy, (bx * bx + by * by) - (dx * dx + dy * dy)],
        [cx - dx, cy - dy, (cx * cx + cy * cy) - (dx * dx + dy * dy)]
    ]
    det = (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
           - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
           + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]))
    return det


def point_in_triangle(pt, f):
    vs = f.vertices()
    if len(vs) != 3:
        return False
    a, b, c = vs
    area = orientation(a, b, c)
    area1 = orientation(pt, b, c)
    area2 = orientation(a, pt, c)
    area3 = orientation(a, b, pt)
    return (area >= 0 and area1 >= 0 and area2 >= 0 and area3 >= 0) or \
        (area <= 0 and area1 <= 0 and area2 <= 0 and area3 <= 0)


class DelaunayTriangulation:
    def __init__(self, points):
        self.points = points
        self.dcel = DCEL()
        self.init_super_triangle()

    def init_super_triangle(self):
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        dx = maxx - minx
        dy = maxy - miny
        delta = max(dx, dy) * 100.0
        v1 = Vertex(minx - delta, miny - delta)
        v2 = Vertex(minx + 2 * delta, miny - delta)
        v3 = Vertex(minx, miny + 2 * delta)
        self.dcel.vertices.extend([v1, v2, v3])

        self.super_face = self.dcel.create_face(v1, v2, v3)

    def locate_point_face(self, p):
        for f in self.dcel.faces:
            vs = f.vertices()
            if len(vs) == 3 and point_in_triangle(p, f):
                return f
        return None

    def insert_point(self, p):
        f = self.locate_point_face(p)
        if f is None:
            return

        vs = f.vertices()
        v_p = Vertex(p.x, p.y)
        self.dcel.vertices.append(v_p)

        self.dcel.faces.remove(f)

        f1 = self.dcel.create_face(v_p, vs[0], vs[1])
        f2 = self.dcel.create_face(v_p, vs[1], vs[2])
        f3 = self.dcel.create_face(v_p, vs[2], vs[0])

        self.fix_delaunay_around_point(v_p)

    def fix_delaunay_around_point(self, v_p):
        incident_triangles = self.get_incident_triangles(v_p)

        changed = True
        while changed:
            changed = False
            for f in incident_triangles:
                if len(f.vertices()) != 3:
                    continue
                edges = self.face_edges(f)
                for e in edges:
                    if self.swap_test(e):
                        changed = True
                        incident_triangles = self.get_incident_triangles(v_p)
                        break
                if changed:
                    break

    def swap_test(self, e):
        if e.twin is None:
            return False

        f1 = e.face
        f2 = e.twin.face
        vs1 = f1.vertices()
        vs2 = f2.vertices()

        if len(vs1) != 3 or len(vs2) != 3:
            return False

        shared_edge = set(vs1).intersection(set(vs2))
        if len(shared_edge) != 2:
            return False
        shared = list(shared_edge)
        unique_f1 = [v for v in vs1 if v not in shared]
        unique_f2 = [v for v in vs2 if v not in shared]

        if len(unique_f1) != 1 or len(unique_f2) != 1:
            return False
        p = unique_f1[0]
        d = unique_f2[0]
        a, b = shared

        val = in_circle(p, a, b, d)
        if val > 0:
            self.edge_flip(e)
            return True
        return False

    def edge_flip(self, e):
        # 该示例未完整实现flip细节，仅作为结构说明。
        f1 = e.face
        f2 = e.twin.face

        v_f1 = f1.vertices()
        v_f2 = f2.vertices()

        shared = list(set(v_f1).intersection(set(v_f2)))
        if len(shared) != 2:
            return
        a, b = shared

        p = [v for v in v_f1 if v not in shared][0]
        d = [v for v in v_f2 if v not in shared][0]

        self.dcel.faces.remove(f1)
        self.dcel.faces.remove(f2)

        f_new1 = self.dcel.create_face(p, d, a)
        f_new2 = self.dcel.create_face(d, p, b)

        # 完整的flip需要更新twin, prev, next指针，此处略。
        # 在实际可用的代码中必须实现完整拓扑更新！

    def face_edges(self, f):
        e = f.edge
        edges = []
        start = e
        while True:
            edges.append(e)
            e = e.next
            if e == start:
                break
        return edges

    def get_incident_triangles(self, v):
        result = []
        start_edge = v.incident_edge
        if start_edge is None:
            return result
        e = start_edge
        visited = set()
        while True:
            f = e.face
            if f is not None and f not in visited:
                visited.add(f)
                result.append(f)
            e = e.prev.twin if e.prev and e.prev.twin else None
            if e is None or e == start_edge:
                break
        return result

    def run(self):
        random.shuffle(self.points)
        for p in self.points:
            self.insert_point(p)


if __name__ == "__main__":
    # 构造一些随机点
    points = [Vertex(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(20)]

    dt = DelaunayTriangulation(points)
    dt.run()

    # 打印结果
    print("Faces in final triangulation:")
    for f in dt.dcel.faces:
        print(f)

    # 使用matplotlib绘制
    # 提取非超级三角形的面进行可视化（超级三角形的巨大面会影响可视化）
    # 我们简单地过滤掉包含超级点（坐标极大的点）的面
    # 假设第一个插入的3个点是超级点
    super_pts = dt.dcel.vertices[:3]
    super_set = set(super_pts)

    fig, ax = plt.subplots()
    for f in dt.dcel.faces:
        vs = f.vertices()
        if len(vs) == 3:
            # 判断该三角形是否涉及超级点
            if any(v in super_set for v in vs):
                continue
            x_coords = [v.x for v in vs]
            y_coords = [v.y for v in vs]
            # 闭合多边形
            x_coords.append(vs[0].x)
            y_coords.append(vs[0].y)
            ax.plot(x_coords, y_coords, 'k-')

    # 绘制原始点
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    ax.plot(xs, ys, 'ro', markersize=3)

    ax.set_aspect('equal', adjustable='box')
    plt.title("Delaunay Triangulation (Randomized Incremental)")
    plt.show()
