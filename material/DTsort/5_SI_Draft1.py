import numpy as np
from sklearn.cluster import KMeans
from base.DCEL import DCEL, Vertex
from base.DCEL.geometry import orientation

class DSLocatorDCEL:
    """
    分布敏感的面片行走定位器，基于 DCEL 邻接关系。
    """
    def __init__(self, dcel: DCEL):
        self.dcel = dcel

    def locate(self, point: Vertex, start_face_id=None) -> int:
        # 找起始面
        faces = [f for f in self.dcel.faces if f != self.dcel.outer_face]
        if start_face_id is None:
            face = faces[0]
        else:
            face = next((f for f in faces if f.id == start_face_id), faces[0])
        # 行走搜索
        while True:
            verts = self.dcel.enumerate_vertices(face)  # [Vertex, Vertex, Vertex]
            inside = True
            for i in range(3):
                a, b = verts[i], verts[(i+1)%3]
                if orientation(a, b, point) < 0:
                    inside = False
                    # 找对应半边，跳到相邻面
                    for he in self.dcel.enumerate_half_edges(face):
                        if he.origin == a and he.next.origin == b:
                            face = he.twin.face
                            break
                    break
            if inside:
                return face.id

class SplayNode:
    def __init__(self, key):
        self.key = key
        self.left = self.right = self.parent = None

class SplayTree:
    """简单 Splay 树，用于自调整面片索引访问。"""
    def __init__(self, keys=None):
        self.root = None
        if keys:
            for k in keys:
                self.insert(k)

    def _rotate(self, x: SplayNode):
        p = x.parent; g = p.parent
        if p.left == x:
            p.left = x.right
            if x.right: x.right.parent = p
            x.right = p
        else:
            p.right = x.left
            if x.left: x.left.parent = p
            x.left = p
        p.parent = x; x.parent = g
        if g:
            if g.left == p: g.left = x
            else: g.right = x

    def _splay(self, x: SplayNode):
        while x.parent:
            p = x.parent; g = p.parent
            if not g:
                self._rotate(x)
            elif (g.left == p and p.left == x) or (g.right == p and p.right == x):
                self._rotate(p); self._rotate(x)
            else:
                self._rotate(x); self._rotate(x)
        self.root = x

    def insert(self, key):
        if not self.root:
            self.root = SplayNode(key)
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left: node = node.left
                else:
                    node.left = SplayNode(key); node.left.parent = node
                    self._splay(node.left); break
            elif key > node.key:
                if node.right: node = node.right
                else:
                    node.right = SplayNode(key); node.right.parent = node
                    self._splay(node.right); break
            else:
                self._splay(node); break

    def search(self, key):
        node = self.root; last = None
        while node:
            last = node
            if key < node.key: node = node.left
            elif key > node.key: node = node.right
            else:
                self._splay(node); return node
        if last: self._splay(last)
        return None

class SelfImprovingDelaunayDCEL:
    """
    自我优化的增量插入 + 边翻转 DCEL 实现。
    """
    def __init__(self, n_groups=10, sample_factor=10, random_state=0):
        self.n_groups = n_groups
        self.sample_factor = sample_factor
        self.random_state = random_state

        self.kmeans = None
        self.dcel = None
        self.locator = None
        self.splay_trees = {}
        self.last_face = {}
        self.super_vertices = set()

    def train(self, samples: list):
        # 1) 聚类分组
        all_pts = np.vstack(samples)
        km = KMeans(n_clusters=self.n_groups, random_state=self.random_state).fit(all_pts)
        self.kmeans = km

        # 2) 构造典型点集 V
        n = samples[0].shape[0]
        Nv = n * self.sample_factor
        inds = np.random.RandomState(self.random_state).choice(len(all_pts), Nv, replace=False)
        V = all_pts[inds]

        # 3) 构建初始 DCEL 三角网（超三角）
        min_xy, max_xy = all_pts.min(axis=0), all_pts.max(axis=0)
        dx, dy = max_xy - min_xy
        m = max(dx, dy) * 10
        super_coords = [
            (min_xy[0]-m, min_xy[1]-m),
            (min_xy[0]+2*m, min_xy[1]-m),
            (min_xy[0]-m, min_xy[1]+2*m),
        ]
        self.dcel = DCEL()
        v_super = [self.dcel.add_vertex(x, y) for x,y in super_coords]
        self.super_vertices.update(v_super)
        self.dcel.create_initial_triangle(*v_super)

        # 4) 增量插入 V 集合
        for (x, y) in V:
            v = self.dcel.add_vertex(x, y)
            self.dcel.insert_point_with_certificate(v)

        # 5) 构建 DSLocator 与 SplayTrees
        self.locator = DSLocatorDCEL(self.dcel)
        B_counts = {g: {} for g in range(self.n_groups)}
        for sample in samples:
            labels = km.predict(sample)
            for (x, y), g in zip(sample, labels):
                pt = Vertex(x, y)
                fid = self.locator.locate(pt)
                B_counts[g][fid] = B_counts[g].get(fid, 0) + 1

        for g, counts in B_counts.items():
            self.splay_trees[g] = SplayTree(keys=list(counts.keys()))

    def run(self, I: np.ndarray):
        # 增量插入 I 中的每个点
        for x, y in I:
            # 1) 添加点
            v = self.dcel.add_vertex(x, y)
            # 2) 分组 & 定位
            g = self.kmeans.predict([[x, y]])[0]
            start_fid = self.last_face.get(g)
            fid = self.locator.locate(v, start_fid)
            self.splay_trees[g].search(fid)
            self.last_face[g] = fid
            # 3) 插入并翻转
            self.dcel.insert_point_with_certificate(v)

        # 提取 Del(I)：忽略含超三角顶点的面
        tris = []
        for face in self.dcel.faces:
            if face == self.dcel.outer_face:
                continue
            verts = self.dcel.enumerate_vertices(face)
            if any(v in self.super_vertices for v in verts):
                continue
            tris.append([(v.x, v.y) for v in verts])
        return tris

if __name__ == "__main__":
    # 示例测试
    import numpy as np
    np.random.seed(0)
    n = 50
    samples = [np.random.rand(n, 2) for _ in range(10)]
    sidt = SelfImprovingDelaunayDCEL(n_groups=4, sample_factor=4)
    sidt.train(samples)
    I = np.random.rand(n, 2)
    tris = sidt.run(I)
    print(f"Del(I) 共 {len(tris)} 个三角形")
