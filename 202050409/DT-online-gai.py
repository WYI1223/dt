# delaunay_numpy_matplotlib.py
# ==========================================
"""
Pure‑NumPy Delaunay Triangulation
--------------------------------
依赖:  numpy  (>=1.17)
       matplotlib (仅在调用 plot() 时需要)

示例:
    import numpy as np, matplotlib.pyplot as plt
    pts = np.random.rand(40, 2)*10         # 40 random 2‑D points
    dt  = DelaunayTriangulation(*pts)      # 构网

    fig, ax = plt.subplots()
    dt.plot(ax=ax, show_points=True)       # 可视化
    plt.show()
"""
from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple

# ---------- 精度与常量 ----------
PI   = np.pi
ev   = np.exp(1) ** PI / 1_000_000_000
ev2  = ev ** 2
INF  = 333                         # “无穷远”坐标

# ---------- 几何工具 ----------
def _left(p, q, r) -> float:
    """ToLeft >0: r 在 pq 左侧；<0: 右侧；≈0 共线"""
    return (p[0]*q[1]-p[1]*q[0] +
            q[0]*r[1]-q[1]*r[0] +
            r[0]*p[1]-r[1]*p[0])

def _to_left(p, q, r):
    a = _left(p, q, r)
    return 0 if abs(a) < ev else a

def _in_triangle(a,b,c,d)->bool:
    t1=_to_left(a,b,d); t2=_to_left(b,c,d); t3=_to_left(c,a,d)
    if t1==0: return (t2<=0 and t3<=0) or (t2>=0 and t3>=0)
    if t1>0 : return t2>=0 and t3>=0
    return t2<=0 and t3<=0

def _in_circle(a, b, c, d) -> bool:
    """返回 d 是否在经过 a,b,c 的外接圆内部（忽略误差 ev）"""
    orient = _to_left(a, b, c)
    if orient == 0:            # 共线 → 无外接圆
        return False

    # 将坐标平移，使 d 成为原点，提升数值稳定性
    def sh(p):
        x, y = p[0] - d[0], p[1] - d[1]
        return [x, y, x*x + y*y]

    det = np.linalg.det([sh(a), sh(b), sh(c)])
    # 根据 orient 修正符号
    return det * orient > ev


# ---------- DCEL 数据结构 ----------
class Vert:   # 顶点
    def __init__(self, p): self.p, self.he = p, None

class HE:     # 半边
    __slots__=("s","e","twin","face","pre","suc","v")
    def __init__(self,s:Vert,e:Vert):
        self.s, self.e = s, e   # 起点/终点
        self.twin=self.face=self.pre=self.suc=None
        self.v=False            # 访问标记

class Face:   # 三角形面
    __slots__=("he","bucket","v")
    def __init__(self,he:HE):
        self.he=he; self.bucket=[]; self.v=False

# ---------- 初始化包含“无穷远”三角形 ----------
def _init_inf_net(points:Sequence[np.ndarray])->Face:
    v1,v2,v3=[Vert(np.array(pt)) for pt in
             ([ INF, 0,0],[0, INF,0],[-INF,-INF,0])]
    h1,h2,h3=[HE(*e) for e in ((v1,v2),(v2,v3),(v3,v1))]
    for v,h in zip((v1,v2,v3),(h1,h2,h3)): v.he=h
    h1.suc,h1.pre = h2,h3
    h2.suc,h2.pre = h3,h1
    h3.suc,h3.pre = h1,h2
    F=Face(h1)
    for h in (h1,h2,h3): h.face=F
    F.bucket=list(points)
    return F

# ---------- 边翻转 ----------
def _edge_flip(h:HE):
    F,G=h.face,h.twin.face
    v1,v2,v3,v4= h.s, h.twin.suc.e, h.e, h.suc.e
    e1,e2,e3,e4= h.twin.suc, h.twin.pre, h.suc, h.pre
    v1.he,e1.face = e1, F
    v3.he,e3.face = e3, F
    v2.he,e2.face = e2, G
    v4.he,e4.face = e4, G
    # 新对角
    e5,e6=HE(v2,v4),HE(v4,v2)
    e5.twin=e6; e6.twin=e5
    # 重新串联
    e1.pre,e1.suc = e4,e5
    e4.pre,e4.suc = e5,e1
    e5.pre,e5.suc = e1,e4
    e2.pre,e2.suc = e6,e3
    e3.pre,e3.suc = e2,e6
    e6.pre,e6.suc = e3,e2
    # 重新指向面
    for he in (e1,e4,e5): he.face=F
    for he in (e2,e3,e6): he.face=G
    # 把点重新分给 F/G
    merged=F.bucket+G.bucket
    F.bucket.clear(); G.bucket.clear()
    for p in merged:
        (F.bucket if _in_triangle(v1.p,v2.p,v4.p,p) else G.bucket).append(p)

# ---------- 面撕裂 ----------
def _clip(face:Face, vo:Vert):
    h1,h2,h3=face.he,face.he.suc,face.he.suc.suc
    def _new(h):
        a=HE(vo,h.s); b=HE(h.e,vo)
        h.pre,h.suc = a,b
        a.pre,b.pre = b,h
        a.suc,b.suc = h,a
        return a,b
    p1,q1=_new(h1); p2,q2=_new(h2); p3,q3=_new(h3)
    # 同步 twin
    p1.twin,q3.twin = q3,p1
    p2.twin,q1.twin = q1,p2
    p3.twin,q2.twin = q2,p3
    # 建三张新面
    F1,F2,F3=[Face(h) for h in (h1,h2,h3)]
    for F,edges in ((F1,(h1,p1,q1)),(F2,(h2,p2,q2)),(F3,(h3,p3,q3))):
        for e in edges: e.face=F
    vo.he=p1
    # 剩余点分桶
    pts=face.bucket; face.bucket=[]   # 原桶
    for p in pts:
        (F1.bucket if _in_triangle(h1.s.p,h2.s.p,vo.p,p)
         else F2.bucket if _in_triangle(h2.s.p,h3.s.p,vo.p,p)
         else F3.bucket).append(p)
    return F1,F2,F3

# ---------- 构网 ----------
def _construct(points) -> Face:
    net = _init_inf_net(points)
    buckets = [net]                      # 待处理面

    while buckets:
        B = buckets.pop()
        if not B.bucket:                 # 空桶跳过
            continue
        p  = B.bucket.pop()
        vo, F = Vert(p), B               # <= 这里上一轮已改好

        _clip(F, vo)                     # 撕裂原面

        # --------- 局部合法化 ----------
        for h in (F.he, F.he.suc, F.he.suc.suc):
            stack = [h]
            while stack:
                e = stack.pop()
                if e.twin and _in_circle(vo.p,
                                         e.twin.s.p, e.twin.e.p,
                                         e.twin.suc.e.p):
                    # ⇩⇩⇩ 只在列表里才删除，避免 ValueError
                    if e.twin.face.bucket and e.twin.face in buckets:
                        buckets.remove(e.twin.face)
                    stack.extend([e.twin.pre, e.twin.suc])
                    _edge_flip(e)

        # --------- 把环绕 vo 的新面重新压栈 ----------
        start = vo.he
        cur   = start.twin.suc
        while True:
            if cur.face.bucket and cur.face not in buckets:
                buckets.append(cur.face)
            cur = cur.twin.suc
            if cur is start:
                break
    return net


# ---------- 遍历 ----------
def _edges(net:Face)->List[Tuple[np.ndarray,np.ndarray]]:
    edges,vis_flag= [],not net.v
    stack=[net]; net.v=vis_flag
    while stack:
        F=stack.pop()
        for h in (F.he,F.he.suc,F.he.suc.suc):
            if h.v!=vis_flag:
                h.v=vis_flag
                if all(abs(c)!=INF for c in (*h.s.p[:2],*h.e.p[:2])):
                    edges.append((h.s.p[:2], h.e.p[:2]))
                if h.twin and h.twin.face.v!=vis_flag:
                    h.twin.face.v=vis_flag
                    stack.append(h.twin.face)
    return edges

def _triangles(net:Face)->List[np.ndarray]:
    tris,vis_flag= [],not net.v
    stack=[net]; net.v=vis_flag
    while stack:
        F=stack.pop()
        h=F.he
        tris.append(np.vstack((h.s.p[:2], h.suc.s.p[:2], h.suc.suc.s.p[:2])))
        for he in (h,h.suc,h.suc.suc):
            if he.twin and he.twin.face.v!=vis_flag:
                he.twin.face.v=vis_flag
                stack.append(he.twin.face)
    return tris

# ---------- 对外类 ----------
class DelaunayTriangulation:
    def __init__(self, *points):
        # points 可能是 tuple/list/ndarray，都转成 ndarray
        def to_xyz(p):
            a = np.asarray(p, dtype=float).ravel()
            if a.size == 2:
                a = np.concatenate([a, [0.0]])
            return a
        pts_xyz = [to_xyz(p) for p in points]

        self._pts = np.array([p[:2] for p in pts_xyz])   # 仅二维坐标保存
        self._net = _construct(pts_xyz)

    # 基本查询 -------------------------------------------------
    def edge_list(self):      return _edges(self._net)
    def triangle_list(self):  return _triangles(self._net)

    # 可视化 ---------------------------------------------------
    def plot(self, ax=None, show_points=False, **line_kw):
        """
        在 matplotlib 轴上绘制德劳内边；可选同时显示散点
        """
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()

        edges = self.edge_list()
        # --- 画边 -------------------------------------------------
        for p, q in edges:
            ax.plot([p[0], q[0]], [p[1], q[1]], **line_kw)

        # --- 画点 -------------------------------------------------
        if show_points:
            if edges:  # 正常情况：从边端点收集
                pts = np.unique(np.vstack([*edges]), axis=0)
            else:  # 还没有边 ⇒ 直接用输入点
                pts = self._pts
            ax.scatter(pts[:, 0], pts[:, 1], s=15, zorder=3)

        ax.set_aspect("equal", adjustable="datalim")

    # 动态插点 -------------------------------------------------
    def insert(self, point:Sequence[float]):
        p=np.asarray(point+(0,)) if len(point)==2 else np.asarray(point)
        # 找位置
        F=_locate_face(p,self._net)
        F.bucket.append(p); _construct([])   # 只触发一次局部流程
        return self

# ——（可选）辅助：定位点所在面，用于 insert -----------
def _locate_face(p, net):
    vis_flag=not net.v; stack=[net]; net.v=vis_flag
    while stack:
        F=stack.pop()
        h=F.he; a,b,c=h.s.p,h.suc.s.p,h.suc.suc.s.p
        if _in_triangle(a,b,c,p): return F
        for he in (h,h.suc,h.suc.suc):
            if he.twin and he.twin.face.v!=vis_flag:
                he.twin.face.v=vis_flag
                stack.append(he.twin.face)
    raise RuntimeError("point outside convex hull?")

if __name__ == "__main__":
    import numpy as np, matplotlib.pyplot as plt
    np.random.seed(0)
    pts = np.random.rand(30,2)*10
    dt  = DelaunayTriangulation(*pts)
    fig, ax = plt.subplots(figsize=(6,6))
    dt.plot(ax, show_points=True, linewidth=1)
    plt.show()
