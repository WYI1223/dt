import time
import numpy as np
from scipy.spatial import Delaunay

class HalvingSplitTree:
    def __init__(self, points, indices=None, depth=0):
        if indices is None:
            indices = list(range(len(points)))
        self.axis = depth % 2
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
        out = []
        if self.index is not None:
            out.append(self.index)
            if self.left:
                out.extend(self.left.preorder())
            if self.right:
                out.extend(self.right.preorder())
        return out

class TrieNode:
    def __init__(self):
        self.children = {}
        self.B = None
        self.Pi = None

class SelfImprovingDelaunay:
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

        for i in range(n):
            for j in range(i+1, n):
                if all(abs(sample[i,0] - sample[j,0]) < 1e-8 and abs(sample[i,1] - sample[j,1]) < 1e-8 for sample in samples):
                    union(i, j)
        for i in range(n):
            for j in range(i+1, n):
                diffs_x = [sample[j,0] - sample[i,0] for sample in samples]
                diffs_y = [sample[j,1] - sample[i,1] for sample in samples]
                if (abs(np.mean(diffs_x)) < 1e-8 and np.std(diffs_x) < 1e-8 and
                    abs(np.mean(diffs_y)) < 1e-8 and np.std(diffs_y) < 1e-8):
                    union(i, j)
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
                        ]))
                        for sample in samples
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
            B = self._compute_B(subset)
            Pi = self._compute_Pi(subset)
            for p in subset:
                # trivial local partition
                tri = self._subproblem_triangulation([p], Pi)
                subtris.append((p.reshape(1,2), tri))
        return self._global_merge(subtris)

    def _retrieve_groups(self, points):
        return {0: points}

    def _local_bucket_partition(self, points, B):
        return [[p] for p in points]

    def _subproblem_triangulation(self, subset_points, Pi):
        return Delaunay(np.array(subset_points)) if len(subset_points) >= 3 else None

    def _global_merge(self, subtris):
        all_pts = np.vstack([b for b, t in subtris])
        return Delaunay(all_pts)

# Measure performance
ns = [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
sample_count = 10
train_times = {}
exec_times = {}

for n in ns:
    samples = [np.random.rand(n, 2) for _ in range(sample_count)]
    sid = SelfImprovingDelaunay()
    if n <= 100:
        t0 = time.perf_counter()
        sid.train(samples)
        train_times[n] = time.perf_counter() - t0
    else:
        train_times[n] = None
        sid.build_net(samples)
    new_pts = np.random.rand(n, 2)
    t0 = time.perf_counter()
    sid.operation(new_pts)
    exec_times[n] = time.perf_counter() - t0

# Estimate training times for n>100 via O(n^3)
t_ref = train_times[100]
est_train_times = {n: (train_times[n] if train_times[n] is not None else t_ref * (n / 100) ** 3) for n in ns}

import matplotlib.pyplot as plt

plt.figure()
plt.plot(ns, [est_train_times[n] for n in ns], label='Training')
plt.plot(ns, [exec_times[n] for n in ns], label='Execution')
plt.xlabel('Number of points (n)')
plt.ylabel('Time (seconds)')
plt.title('Performance of SelfImprovingDelaunay')
plt.legend()
plt.show()

for n in ns:
    tt = est_train_times[n]
    te = exec_times[n]
    if train_times[n] is None:
        print(f"n={n}: Training time ≈ {tt:.4f}s (estimated), Execution time = {te:.4f}s")
    else:
        print(f"n={n}: Training time = {tt:.4f}s, Execution time = {te:.4f}s")

