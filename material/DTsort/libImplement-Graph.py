import numpy as np
from scipy.spatial import Delaunay
from collections import deque
import matplotlib.pyplot as plt

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

class TrieNode:
    """
    Trie node storing children for encoding group queries,
    and holding B and Pi structures for that prefix.
    """
    def __init__(self):
        self.children = {}
        self.B = None
        self.Pi = None

class SelfImprovingDelaunay:
    """
    Implements the self-improving Delaunay triangulation algorithm
    with training and operation phases.
    """
    def __init__(self):
        self.trie_roots = {}
        self.V = None
        self.V_delaunay = None
        self.big_triangle = None

    def train(self, samples):
        groups = self.approximate_partition(samples)
        self.build_net(samples)
        self.build_tries(samples, groups)

    def approximate_partition(self, samples):
        """
        Approximate the hidden partition via dependency tests.
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

        # 1. Constant tests
        for i in range(n):
            for j in range(i+1, n):
                const_x = all(abs(sample[i,0] - sample[j,0]) < 1e-8 for sample in samples)
                const_y = all(abs(sample[i,1] - sample[j,1]) < 1e-8 for sample in samples)
                if const_x and const_y:
                    union(i, j)
        # 2. Bivariate tests
        for i in range(n):
            for j in range(i+1, n):
                diffs_x = [sample[j,0] - sample[i,0] for sample in samples]
                diffs_y = [sample[j,1] - sample[i,1] for sample in samples]
                if (abs(np.mean(diffs_x)) < 1e-8 and np.std(diffs_x) < 1e-8 and
                    abs(np.mean(diffs_y)) < 1e-8 and np.std(diffs_y) < 1e-8):
                    union(i, j)
        # 3. Trivariate tests
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if find(i) != find(j) and find(j) != find(k) and find(i) != find(k):
                        continue
                    if any(
                        (abs(sample[i,0]-sample[j,0])<1e-8 and abs(sample[i,1]-sample[j,1])<1e-8) or
                        (abs(sample[j,0]-sample[k,0])<1e-8 and abs(sample[j,1]-sample[k,1])<1e-8) or
                        (abs(sample[i,0]-sample[k,0])<1e-8 and abs(sample[i,1]-sample[k,1])<1e-8)
                        for sample in samples
                    ):
                        continue
                    dets = [
                        np.linalg.det(np.array([
                            [sample[i,0], sample[i,1], 1],
                            [sample[j,0], sample[j,1], 1],
                            [sample[k,0], sample[k,1], 1]
                        ])) for sample in samples
                    ]
                    if all(abs(d) < 1e-6 for d in dets):
                        union(i, j)
                        union(j, k)

        comps = {}
        for idx in range(n):
            root = find(idx)
            comps.setdefault(root, []).append(idx)
        return {gid: comp for gid, comp in enumerate(sorted(comps.values(), key=lambda x: x[0]))}

    def build_net(self, samples):
        pts = np.vstack(samples)
        n = samples[0].shape[0]
        m = int(n * np.log(n) + 1)
        chosen = np.random.choice(len(pts), size=min(m, len(pts)), replace=False)
        self.V = pts[chosen]
        maxc, minc = np.max(self.V, axis=0), np.min(self.V, axis=0)
        d = max(maxc - minc) * 10
        p1 = minc + np.array([-1, -1]) * d
        p2 = minc + np.array([3, -1]) * d
        p3 = minc + np.array([-1, 3]) * d
        self.big_triangle = np.vstack([p1, p2, p3])
        self.V = np.vstack([self.V, self.big_triangle])
        self.V_delaunay = Delaunay(self.V)

    def build_tries(self, samples, groups):
        for gid, idxs in groups.items():
            root = TrieNode()
            for sample in samples:
                key = tuple(sorted(idxs))
                node = root
                for k in key:
                    node = node.children.setdefault(k, TrieNode())
                node.B = self._compute_B(sample[idxs])
                node.Pi = self._compute_Pi(sample[idxs])
            self.trie_roots[gid] = root

    def _compute_B(self, points):
        return self.V_delaunay.find_simplex(points).tolist()

    def _compute_Pi(self, points):
        return HalvingSplitTree(points)

    def operation(self, points):
        groups = self._retrieve_groups(points)
        subtris = []
        for gid, subset in groups.items():
            B = self._query_B(gid, subset)
            Pi = self._query_Pi(gid, subset)
            buckets = self._local_bucket_partition(subset, B)
            for buck in buckets:
                tri = self._subproblem_triangulation(buck, Pi)
                subtris.append((buck, tri))
        return self._global_merge(subtris)

    def _retrieve_groups(self, points):
        # placeholder grouping for operation phase
        return {0: points}

    def _query_B(self, gid, points):
        return self._compute_B(points)

    def _query_Pi(self, gid, points):
        return self._compute_Pi(points)

    def _local_bucket_partition(self, points, B):
        # simple bucketing: each point its own bucket
        return [[p] for p in points]

    def _subproblem_triangulation(self, subset_points, Pi):
        return Delaunay(np.array(subset_points)) if len(subset_points) >= 3 else None

    def _global_merge(self, subtris):
        all_pts = np.vstack([b for b, t in subtris])
        return Delaunay(all_pts)

    def plot_triangulation(self, delaunay, show=True, ax=None, **plot_kwargs):
        """Plot a Delaunay triangulation using matplotlib."""
        if ax is None:
            fig, ax = plt.subplots()
        pts = delaunay.points
        for simplex in delaunay.simplices:
            tri = pts[simplex]
            tri = np.vstack([tri, tri[0]])
            ax.plot(tri[:,0], tri[:,1], **plot_kwargs)
        ax.plot(pts[:,0], pts[:,1], 'o')
        if show:
            plt.show()
        return ax

if __name__ == "__main__":
    n = 100
    samples = [np.random.rand(n,2) for _ in range(10)]
    sid = SelfImprovingDelaunay()
    sid.train(samples)
    new_pts = np.random.rand(n,2)
    result = sid.operation(new_pts)
    if result:
        print("Triangle count:", len(result.simplices))
        sid.plot_triangulation(result)
    else:
        print("No triangulation result to plot.")
