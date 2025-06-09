import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from collections import deque

class HalvingSplitTree:
    """
    Binary tree that splits points by median alternately on x and y to achieve roughly balanced partition.
    Stores the splitting axis, point index at split, and left/right subtrees.
    """
    def __init__(self, points, indices=None, depth=0):
        if indices is None:
            indices = list(range(len(points)))
        self.axis = depth % 2  # 0: x, 1: y
        if len(indices) <= 1:
            self.index = indices[0] if indices else None
            self.left = None
            self.right = None
            return
        # sort and split
        indices.sort(key=lambda i: points[i][self.axis])
        mid = len(indices) // 2
        self.index = indices[mid]
        self.left = HalvingSplitTree(points, indices[:mid], depth + 1)
        self.right = HalvingSplitTree(points, indices[mid+1:], depth + 1)

    def preorder(self):
        """Return indices in preorder traversal."""
        out = []
        if self.index is not None:
            out.append(self.index)
            if self.left:
                out.extend(self.left.preorder())
            if self.right:
                out.extend(self.right.preorder())
        return out

class SelfImprovingDelaunay:
    """
    Implements the self-improving Delaunay triangulation algorithm
    with training and operation phases.
    """
    def __init__(self):
        # grouping information from training
        self.groups = {}
        # net V and its Delaunay triangulation
        self.V = None
        self.V_delaunay = None
        self.big_triangle = None

    def train(self, samples):
        # 1. approximate hidden partition into groups
        groups = self.approximate_partition(samples)
        self.groups = groups
        # 2. build epsilon-net (skeleton) over V
        self.build_net(samples)
        # TODO: (optional) build distribution-sensitive tries for B and Pi

    def approximate_partition(self, samples):
        """
        Approximate the hidden partition via
        1. constant tests
        2. bivariate tests
        3. trivariate tests
        """
        n = samples[0].shape[0]
        parent = list(range(n))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        def union(u, v):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru
        # 1. constant tests
        for i in range(n):
            for j in range(i+1, n):
                const_x = all(abs(s[i,0] - s[j,0]) < 1e-8 for s in samples)
                const_y = all(abs(s[i,1] - s[j,1]) < 1e-8 for s in samples)
                if const_x and const_y:
                    union(i, j)
        # 2. bivariate tests
        for i in range(n):
            for j in range(i+1, n):
                dx = [s[j,0] - s[i,0] for s in samples]
                dy = [s[j,1] - s[i,1] for s in samples]
                if (abs(np.mean(dx))<1e-8 and np.std(dx)<1e-8 and
                    abs(np.mean(dy))<1e-8 and np.std(dy)<1e-8):
                    union(i, j)
        # 3. trivariate tests
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # must share at least one unioned pair
                    if find(i)!=find(j) and find(j)!=find(k) and find(i)!=find(k):
                        continue
                    # skip identical points
                    if any((abs(s[i,0]-s[j,0])<1e-8 and abs(s[i,1]-s[j,1])<1e-8) or
                           (abs(s[j,0]-s[k,0])<1e-8 and abs(s[j,1]-s[k,1])<1e-8) or
                           (abs(s[i,0]-s[k,0])<1e-8 and abs(s[i,1]-s[k,1])<1e-8)
                           for s in samples):
                        continue
                    dets = [np.linalg.det(np.array([
                                [s[i,0], s[i,1], 1],
                                [s[j,0], s[j,1], 1],
                                [s[k,0], s[k,1], 1]
                            ])) for s in samples]
                    if all(abs(d)<1e-6 for d in dets):
                        union(i, j)
                        union(j, k)
        # collect groups
        comps = {}
        for idx in range(n):
            root = find(idx)
            comps.setdefault(root, []).append(idx)
        # normalize to gid->list
        return {gid: comp for gid, comp in enumerate(sorted(comps.values(), key=lambda x: x[0]))}

    def build_net(self, samples):
        """
        Build the epsilon-net V by sampling and add a big enclosing triangle.
        """
        pts = np.vstack(samples)
        n = samples[0].shape[0]
        m = int(n * np.log(n) + 1)
        chosen = np.random.choice(len(pts), size=min(m, len(pts)), replace=False)
        self.V = pts[chosen]
        # build a big triangle that encloses all V
        maxc, minc = np.max(self.V, axis=0), np.min(self.V, axis=0)
        d = max(maxc - minc) * 10
        p1 = minc + np.array([-1, -1]) * d
        p2 = minc + np.array([ 3, -1]) * d
        p3 = minc + np.array([-1,  3]) * d
        self.big_triangle = np.vstack([p1, p2, p3])
        self.V = np.vstack([self.V, self.big_triangle])
        self.V_delaunay = Delaunay(self.V)

    @staticmethod
    def _incircle(a, b, c, p):
        """Return True if p lies inside circumcircle of triangle (a,b,c) (assumes CCW)."""
        M = np.array([
            [a[0]-p[0], a[1]-p[1], (a[0]-p[0])**2 + (a[1]-p[1])**2],
            [b[0]-p[0], b[1]-p[1], (b[0]-p[0])**2 + (b[1]-p[1])**2],
            [c[0]-p[0], c[1]-p[1], (c[0]-p[0])**2 + (c[1]-p[1])**2]
        ])
        return np.linalg.det(M) > 0

    def _collect_cluster(self, simplex_id, point):
        """Collect all simplices whose circumcircle contains 'point' via BFS in dual graph."""
        cluster = set([simplex_id])
        queue = deque([simplex_id])
        neighbors = self.V_delaunay.neighbors
        while queue:
            s = queue.popleft()
            for nbr in neighbors[s]:
                if nbr != -1 and nbr not in cluster:
                    tri_pts = self.V[self.V_delaunay.simplices[nbr]]
                    if SelfImprovingDelaunay._incircle(tri_pts[0], tri_pts[1], tri_pts[2], point):
                        cluster.add(nbr)
                        queue.append(nbr)
        return cluster

    def operation(self, points):
        # 1. split input into groups
        groups = self._retrieve_groups(points)
        subtris = []
        # 2. for each group handle local triangulations
        for gid, subset in groups.items():
            subset = np.asarray(subset)
            # skip small groups
            if subset.shape[0] < 3:
                continue
            # locate each point in the epsilon-net triangulation
            B = self.V_delaunay.find_simplex(subset)
            # build split tree on the points of this group
            Pi = HalvingSplitTree(subset)
            # partition into local buckets by circumcircle influence
            buckets = self._local_bucket_partition(subset, B)
            # triangulate each bucket if large enough
            for buck in buckets:
                if len(buck) < 3:
                    continue
                tri = self._subproblem_triangulation(buck, Pi)
                if tri is not None:
                    subtris.append((buck, tri))
        # 3. if no local triangulations, fallback to global Delaunay
        if not subtris:
            return Delaunay(points)
        # 4. merge all local triangulations into final
        return self._global_merge(subtris)

    def _retrieve_groups(self, points):
        """Assign each input point to its group via stored self.groups indices."""
        return {gid: points[idxs] for gid, idxs in self.groups.items()}

    def _local_bucket_partition(self, subset, B):
        """
        Partition subset points into buckets Q_{j,t} for each affected simplex t.
        subset: (m,2) array, B: length-m array of simplex IDs.
        """
        buckets = {}
        for i, simplex in enumerate(B):
            if simplex < 0:
                continue
            cluster = self._collect_cluster(simplex, subset[i])
            for t in cluster:
                buckets.setdefault(t, []).append(subset[i])
        return list(buckets.values())

    def _subproblem_triangulation(self, subset_points, Pi):
        """
        Perform Delaunay triangulation on subset_points, using Pi's preorder to sequence input.
        """
        pts = np.array(subset_points)
        if len(pts) < 3:
            return None
        order = Pi.preorder()
        # filter and reorder
        order = [i for i in order if i < len(pts)]
        ordered = pts[order]
        return Delaunay(ordered)

    def _global_merge(self, subtris):
        """Merge all local triangulations by recomputing Delaunay on union."""
        if not subtris:
            return None
        all_pts = np.vstack([b for b, _ in subtris])
        return Delaunay(all_pts)



def plot_delaunay(points, delaunay, ax=None, show=True,


                  point_style='o', line_style='k-', title=None):
    """
    绘制 Delaunay 三角网。

    Parameters
    ----------
    points : (n,2) ndarray
        原始点集；应该和 ``delaunay.points`` 对应。
    delaunay : scipy.spatial.Delaunay
        要绘制的三角剖分结果。
    ax : matplotlib.axes.Axes, optional
        若给定则在现有坐标轴上作图；否则自动创建新图窗。
    show : bool, default True
        调用结束后是否 ``plt.show()``。
    point_style : str, default 'o'
        点的 matplotlib 样式字符串。
    line_style : str, default 'k-'
        边的 matplotlib 样式字符串。
    title : str or None
        图标题。
    """
    if ax is None:  # → 创建画布
        _, ax = plt.subplots(figsize=(6, 6))

    # 三角边
    ax.triplot(points[:, 0], points[:, 1],
               delaunay.simplices.copy(), line_style, linewidth=.8)
    # 点
    ax.plot(points[:, 0], points[:, 1], point_style, markersize=3)

    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if show:
        plt.show()

    return ax

if __name__ == "__main__":
    # example usage and tests
    n = 100
    samples = [np.random.rand(n,2) for _ in range(10)]
    sid = SelfImprovingDelaunay()
    sid.train(samples)
    new_pts = np.random.rand(n,2)
    result = sid.operation(new_pts)
    print("Triangle count:", len(result.simplices) if result else 0)
    assert isinstance(result, Delaunay) and len(result.simplices) > 0, \
        f"Expected non-empty triangulation, got {result}"

    # tests for approximate_partition
    pts_const = np.tile([1.0, 2.0], (5,1))
    samples_const = [pts_const for _ in range(3)]
    groups_const = sid.approximate_partition(samples_const)
    assert len(groups_const) == 1 and sorted(groups_const[0]) == list(range(5)), \
        f"Expected single group, got {groups_const}"

    pts_clusters = np.array([[0,0],[0,0],[1,1],[1,1]])
    samples_clusters = [pts_clusters for _ in range(4)]
    groups_clust = sid.approximate_partition(samples_clusters)
    sorted_comps = sorted([sorted(v) for v in groups_clust.values()])
    assert sorted_comps == [[0,1],[2,3]], f"Expected [[0,1],[2,3]], got {sorted_comps}"

    pts_three = np.array([[2,2],[2,2],[3,3]])
    samples_three = [pts_three for _ in range(3)]
    groups_three = sid.approximate_partition(samples_three)
    sorted_three = sorted([sorted(v) for v in groups_three.values()])
    assert sorted_three == [[0,1],[2]], f"Expected [[0,1],[2]], got {sorted_three}"

    pts_linear = np.array([[0,0],[1,1],[2,2],[3,3]])
    samples_linear = [pts_linear for _ in range(2)]
    groups_linear = sid.approximate_partition(samples_linear)
    sorted_linear = sorted([sorted(v) for v in groups_linear.values()])
    assert sorted_linear == [[0],[1],[2],[3]], f"Expected [[0],[1],[2],[3]], got {sorted_linear}"

    print("All tests passed.")

    print("Triangle count:", len(result.simplices) if result else 0)

    # 可视化：把 new_pts 和 result 传进去
    if result is not None:
        plot_delaunay(new_pts, result, title="Self-Improving Delaunay Triangulation")

