"""
注：
    1. 主要用来求德劳内三角剖分，算法的思路可参考：https://www.bilibili.com/video/BV1Ck4y1z7VT
    2. 时间复杂度 O(n log n)，一般情况足够使用
    3. 只需导入 DelaunayTrianglation 类
"""

import numpy as np
import time
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.constants import PI
from manimlib.utils.config_ops import digest_config
from manimlib.mobject.geometry import Dot, Line, Polygon
from manimlib.scene.scene import Scene

# 误差控制
ev = np.exp(1)**PI/1000000000
ev_sq = ev**2

# 无穷大常量
Infinity = 333

# 判断两个点是否相等
def point_is_equal(p, q):
    p, q = np.array(p), np.array(q)
    return np.dot(q-p, q-p) < ev_sq

# 二维叉积，用于方向判断
def cross2(p, q, b):
    return (p[0]*q[1] - p[1]*q[0]
          + q[0]*b[1] - q[1]*b[0]
          + b[0]*p[1] - b[1]*p[0])

# 忽略误差，b 在向量 pq 左侧则为正，右侧为负
def ToLeft(p, q, b):
    a = cross2(p, q, b)
    if abs(a) < ev:
        return 0
    return a

# 判断点 d 是否在三角形 pqb 内部或边上
def InTriangle(p, q, b, d):
    tl1 = ToLeft(p, q, d)
    if abs(tl1) < ev:
        tl2 = ToLeft(q, b, d)
        tl3 = ToLeft(b, p, d)
        return (tl2 < ev and tl3 < ev) or (tl2 > -ev and tl3 > -ev)
    if tl1 > ev:
        return ToLeft(q, b, d) > -ev and ToLeft(b, p, d) > -ev
    return ToLeft(q, b, d) < ev and ToLeft(b, p, d) < ev

# 判断点 d 是否在由 p,q,b 三点确定的外接圆内
def InCircle(p, q, b, d):
    a13 = p[0]**2 + p[1]**2
    a23 = q[0]**2 + q[1]**2
    a33 = b[0]**2 + b[1]**2
    a43 = d[0]**2 + d[1]**2
    det = np.linalg.det([
        [p[0], p[1], a13, 1],
        [q[0], q[1], a23, 1],
        [b[0], b[1], a33, 1],
        [d[0], d[1], a43, 1],
    ])
    return det < -ev

# 面表示，用于半边结构
class Face:
    def __init__(self, halfedge):
        self.Visit = False
        self.HalfEdge = halfedge
        self.Bucket = None

# 顶点
class Vertice:
    def __init__(self, point):
        self.Point = point
        self.HalfEdge = None

# 半边
class HalfEdge:
    def __init__(self, start, end):
        self.Visit = False
        self.Start = start
        self.End = end
        self.Twin = None
        self.Face = None
        self.Pre = None
        self.Suc = None

# 桶，用于维护待插入点
class Bucket:
    def __init__(self, points):
        self.Points = points
        self.Face = None

# 初始化包含无穷远点的大三角形网
def InitInfNet(points=None):
    infv1 = Vertice(np.array([Infinity, 0, 0]))
    infv2 = Vertice(np.array([0, Infinity, 0]))
    infv3 = Vertice(np.array([-Infinity, -Infinity, 0]))

    he1 = HalfEdge(infv1, infv2)
    he2 = HalfEdge(infv2, infv3)
    he3 = HalfEdge(infv3, infv1)

    infv1.HalfEdge = he1
    infv2.HalfEdge = he2
    infv3.HalfEdge = he3

    face = Face(he1)

    he1.Pre = he3; he1.Suc = he2; he1.Face = face
    he2.Pre = he1; he2.Suc = he3; he2.Face = face
    he3.Pre = he2; he3.Suc = he1; he3.Face = face

    bucket = Bucket(points or [])
    bucket.Face = face
    face.Bucket = bucket

    return face

# 边翻转操作
def EdgeFlipping(halfedge):
    visit = halfedge.Face.Visit
    v1 = halfedge.Start
    v2 = halfedge.Twin.Suc.End
    v3 = halfedge.End
    v4 = halfedge.Suc.End

    e1 = halfedge.Twin.Suc
    e2 = halfedge.Twin.Pre
    e3 = halfedge.Suc
    e4 = halfedge.Pre

    v1.HalfEdge = e1
    v2.HalfEdge = e2
    v3.HalfEdge = e3
    v4.HalfEdge = e4

    oldpts = list(halfedge.Face.Bucket.Points) + list(halfedge.Twin.Face.Bucket.Points)
    pts1, pts2 = [], []
    p1, p2, p4 = v1.Point, v2.Point, v4.Point
    for pt in oldpts:
        if InTriangle(p1, p2, p4, pt): pts1.append(pt)
        else: pts2.append(pt)

    newf1, newf2 = Face(e1), Face(e2)
    newf1.Visit = visit; newf2.Visit = visit

    e5 = HalfEdge(v2, v4); e6 = HalfEdge(v4, v2)
    e5.Twin = e6; e6.Twin = e5
    e5.Visit = visit; e6.Visit = visit

    # 链接 newf1
    e1.Suc = e5; e5.Suc = e4; e4.Suc = e1
    e1.Pre = e4; e4.Pre = e5; e5.Pre = e1
    # 链接 newf2
    e2.Suc = e3; e3.Suc = e6; e6.Suc = e2
    e2.Pre = e6; e6.Pre = e3; e3.Pre = e2

    for e in (e1, e4, e5): e.Face = newf1
    for e in (e2, e3, e6): e.Face = newf2

    b1 = Bucket(pts1); b2 = Bucket(pts2)
    b1.Face = newf1; b2.Face = newf2
    newf1.Bucket = b1; newf2.Bucket = b2

# 撕裂三角形以插入新顶点
def ClipFace(face, vo, pts):
    visit = face.Visit
    hf1, hf2, hf3 = face.HalfEdge, face.HalfEdge.Suc, face.HalfEdge.Suc.Suc

    f1, f2, f3 = Face(hf1), Face(hf2), Face(hf3)
    for f in (f1, f2, f3): f.Visit = visit

    # 对每条边拆分
    new_edges = []
    for hf in (hf1, hf2, hf3):
        pre = HalfEdge(vo, hf.Start)
        suc = HalfEdge(hf.End, vo)
        pre.Visit = visit; suc.Visit = visit
        hf.Pre = pre; hf.Suc = suc
        pre.Pre = suc; pre.Suc = hf
        suc.Pre = hf; suc.Suc = pre
        hf.Face = f1 if hf is hf1 else (f2 if hf is hf2 else f3)
        pre.Face = hf.Face; suc.Face = hf.Face
        new_edges.append((hf, pre, suc))

    vo.HalfEdge = new_edges[0][1]

    # 建立 twin 关系
    new_edges[0][1].Twin = new_edges[2][2]; new_edges[2][2].Twin = new_edges[0][1]
    new_edges[1][1].Twin = new_edges[0][2]; new_edges[0][2].Twin = new_edges[1][1]
    new_edges[2][1].Twin = new_edges[1][2]; new_edges[1][2].Twin = new_edges[2][1]

    # 分配剩余点到子桶
    pvo = vo.Point
    verts = [hf.Start.Point for hf in (hf1, hf2, hf3)]
    buckets = ([], [], [])
    for pt in pts:
        if InTriangle(verts[0], verts[1], pvo, pt):
            buckets[0].append(pt)
        elif InTriangle(verts[1], verts[2], pvo, pt):
            buckets[1].append(pt)
        else:
            buckets[2].append(pt)

    for f, bpts in zip((f1, f2, f3), buckets):
        bucket = Bucket(bpts); bucket.Face = f; f.Bucket = bucket

    return f1, f2, f3

# 遍历网的所有边，输出 DT 边列表
def VisitNet(face):
    visit = face.Visit
    face.Visit = not visit
    stack = [face]
    edges = []

    while stack:
        f = stack.pop()
        e1, e2, e3 = f.HalfEdge, f.HalfEdge.Suc, f.HalfEdge.Suc.Suc
        for e in (e1, e2, e3):
            if not point_is_equal(e.Start.Point, [Infinity, 0, 0]) and not point_is_equal(e.End.Point, [Infinity, 0, 0]):
                if e.Visit == visit:
                    edges.append([e.Start.Point, e.End.Point])
                e.Visit = not visit
                if e.Twin and e.Twin.Face.Visit == visit:
                    e.Twin.Visit = not visit
                    e.Twin.Face.Visit = not visit
                    stack.append(e.Twin.Face)
    return edges

# 可视化三角形网
def VisitTriangles(face):
    visit = face.Visit
    face.Visit = not visit
    stack = [face]
    vgroup = VGroup()

    while stack:
        f = stack.pop()
        e1, e2, e3 = f.HalfEdge, f.HalfEdge.Suc, f.HalfEdge.Suc.Suc
        for e in (e1, e2, e3): e.Visit = not visit
        pts = [e1.Start.Point, e2.Start.Point, e3.Start.Point]
        vgroup.add(Polygon(*pts))
        for e in (e1, e2, e3):
            if e.Twin and e.Twin.Face.Visit == visit:
                e.Twin.Face.Visit = not visit
                stack.append(e.Twin.Face)
    return vgroup

# 构造 Delaunay 三角剖分网
def ConstructNet(points=None):
    root = InitInfNet(points)
    buckets = [root.Bucket]

    while buckets:
        bucket = buckets.pop()
        pt = bucket.Points.pop()
        vo = Vertice(pt)
        f = bucket.Face
        hf1, hf2, hf3 = f.HalfEdge, f.HalfEdge.Suc, f.HalfEdge.Suc.Suc

        new_faces = ClipFace(f, vo, bucket.Points)
        stack_edges = [hf1, hf2, hf3]
        while stack_edges:
            e = stack_edges.pop()
            if e.Twin:
                p1 = vo.Point; p2 = e.Twin.Start.Point; p3 = e.Twin.End.Point; p4 = e.Twin.Suc.End.Point
                if InCircle(p1, p2, p3, p4):
                    tb = e.Twin.Face.Bucket
                    if tb.Points: buckets.remove(tb)
                    stack_edges.extend([e.Twin.Pre, e.Twin.Suc])
                    EdgeFlipping(e)

        # 加入新桶
        ring = vo.HalfEdge
        curr = ring.Twin.Suc
        while curr != ring:
            b = curr.Face.Bucket
            if b.Points: buckets.append(b)
            curr = curr.Twin.Suc
        b = curr.Face.Bucket
        if b.Points: buckets.append(b)

    return root

# DelaunayTrianglation 类，封装 manim 可视化
class DelaunayTrianglation(VGroup):
    def __init__(self, *points, **kwargs):
        digest_config(self, kwargs)
        self.net = ConstructNet(list(points))
        self.kwargs = kwargs
        VGroup.__init__(self, *[Line(*edge, **kwargs) for edge in VisitNet(self.net)])

    def VisitNet(self):
        return VisitNet(self.net)

    def VisitTriangles(self):
        return VisitTriangles(self.net)

    def GetNet(self):
        return self.net

    def InsertPoint(self, point):
        # 增量插入
        # posface = get_point_posface(point, self.net)
        # 此例简化为直接插入
        ConstructNet([*[v for v in []], point])
        self.become(VGroup(*[Line(*edge, **self.kwargs) for edge in VisitNet(self.net)]))
        return self

# 测试类，只演示 Delaunay
class test(Scene):
    def construct(self):
        np.random.seed(2007)
        points = [[
            np.random.randint(-70000, 70000)/10500,
            np.random.randint(-38000, 38000)/10500,
            0
        ] for _ in range(800)]
        dots = [Dot(p).scale(0.5) for p in points]
        self.add(*dots)
        start = time.perf_counter()
        delaunay = DelaunayTrianglation(*points)
        self.add(delaunay)
        end = time.perf_counter()
        print("Delaunay time:", end - start)
        self.wait()
