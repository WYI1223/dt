import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import pandas as pd

# Define HalvingSplitTree and SelfImprovingDelaunay classes (as provided)
class HalvingSplitTree:
    def __init__(self, points, indices=None, depth=0):
        self.points = points
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

class TrieNode:
    def __init__(self):
        self.children = {}
        self.Pi = None

class SelfImprovingDelaunay:
    def __init__(self):
        self.trie_roots = {}
        self.V = None
        self.V_delaunay = None
        self.big_triangle = None
        self.groups = None

    def train(self, samples):
        groups = self.approximate_partition(samples)
        self.groups = groups
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
                    if any((abs(sample[a,0] - sample[b,0]) < 1e-8 and abs(sample[a,1] - sample[b,1]) < 1e-8)
                           for sample in samples for a,b in [(i,j),(j,k),(i,k)]):
                        continue
                    dets = [np.linalg.det(np.array([[sample[i,0], sample[i,1],1],
                                                   [sample[j,0], sample[j,1],1],
                                                   [sample[k,0], sample[k,1],1]]))
                            for sample in samples]
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
        d = np.max(maxc - minc) * 10
        p1 = minc + np.array([-1, -1]) * d
        p2 = minc + np.array([3, -1]) * d
        p3 = minc + np.array([-1, 3]) * d
        self.big_triangle = np.vstack([p1, p2, p3])
        self.V = np.vstack([self.V, self.big_triangle])
        self.V_delaunay = Delaunay(self.V)

    def build_tries(self, samples, groups):
        for gid, idxs in groups.items():
            root = TrieNode()
            merged = np.vstack([sample[idxs] for sample in samples])
            root.Pi = HalvingSplitTree(merged)
            for sample in samples:
                subset = sample[idxs]
                B_seq = tuple(self.V_delaunay.find_simplex(subset).tolist())
                Pi = HalvingSplitTree(subset)
                node = root
                for s in B_seq:
                    node = node.children.setdefault(s, TrieNode())
                node.Pi = Pi
            self.trie_roots[gid] = root

    def operation(self, points):
        groups = self._retrieve_groups(points)
        subtris = []
        for gid, subset in groups.items():
            B_seq = tuple(self.V_delaunay.find_simplex(subset).tolist())
            Pi = self._query_Pi(gid, B_seq)
            buckets = self._local_bucket_partition(subset, Pi)
            for buck in buckets:
                tri = Delaunay(np.array(buck)) if len(buck) >= 3 else None
                subtris.append((buck, tri))
        return self._global_merge(subtris)

    def _retrieve_groups(self, points):
        return {gid: points[idxs] for gid, idxs in self.groups.items()}

    def _query_Pi(self, gid, B_seq):
        node = self.trie_roots[gid]
        for s in B_seq:
            child = node.children.get(s)
            if child is None:
                return node.Pi
            node = child
        return node.Pi or node.Pi

    def _local_bucket_partition(self, subset_points, Pi):
        buckets = {}
        for p in subset_points:
            node = Pi
            while node.left and node.right:
                axis = node.axis
                split_val = node.points[node.index][axis]
                node = node.left if p[axis] <= split_val else node.right
            leaf_id = node.index
            buckets.setdefault(leaf_id, []).append(p)
        return list(buckets.values())

    def _global_merge(self, subtris):
        all_pts = np.vstack([b for b, _ in subtris])
        return Delaunay(all_pts)

# Measure performance
ns = [10, 50, 100, 250, 500]
train_times = []
op_times = []

for n in ns:
    samples = [np.random.rand(n, 2) for _ in range(10)]
    sid = SelfImprovingDelaunay()
    start = time.perf_counter()
    sid.train(samples)
    train_times.append(time.perf_counter() - start)
    pts = np.random.rand(n, 2)
    start = time.perf_counter()
    sid.operation(pts)
    op_times.append(time.perf_counter() - start)

# Create DataFrame and display
df = pd.DataFrame({
    'n': ns,
    'train_time_s': train_times,
    'operation_time_s': op_times
})
print(df)
