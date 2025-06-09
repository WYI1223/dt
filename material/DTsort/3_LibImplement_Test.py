import time
import numpy as np
from scipy.spatial import Delaunay
from collections import deque
import matplotlib.pyplot as plt

# --- Class definitions from provided code ---
class HalvingSplitTree:
    """
    Binary tree that splits points by median alternately on x and y to achieve roughly balanced partition.
    """
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

class SelfImprovingDelaunay:
    def __init__(self):
        self.groups = {}
        self.V = None
        self.V_delaunay = None
        self.big_triangle = None

    def train(self, samples):
        groups = self.approximate_partition(samples)
        self.groups = groups
        self.build_net(samples)

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
        # constant tests
        for i in range(n):
            for j in range(i+1, n):
                const_x = all(abs(s[i,0] - s[j,0]) < 1e-8 for s in samples)
                const_y = all(abs(s[i,1] - s[j,1]) < 1e-8 for s in samples)
                if const_x and const_y:
                    union(i, j)
        # bivariate tests
        for i in range(n):
            for j in range(i+1, n):
                dx = [s[j,0] - s[i,0] for s in samples]
                dy = [s[j,1] - s[i,1] for s in samples]
                if (abs(np.mean(dx))<1e-8 and np.std(dx)<1e-8 and
                    abs(np.mean(dy))<1e-8 and np.std(dy)<1e-8):
                    union(i, j)
        # trivariate tests
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if find(i)!=find(j) and find(j)!=find(k) and find(i)!=find(k):
                        continue
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
        p2 = minc + np.array([ 3, -1]) * d
        p3 = minc + np.array([-1,  3]) * d
        self.big_triangle = np.vstack([p1, p2, p3])
        self.V = np.vstack([self.V, self.big_triangle])
        self.V_delaunay = Delaunay(self.V)

    @staticmethod
    def _incircle(a, b, c, p):
        M = np.array([
            [a[0]-p[0], a[1]-p[1], (a[0]-p[0])**2 + (a[1]-p[1])**2],
            [b[0]-p[0], b[1]-p[1], (b[0]-p[0])**2 + (b[1]-p[1])**2],
            [c[0]-p[0], c[1]-p[1], (c[0]-p[0])**2 + (c[1]-p[1])**2]
        ])
        return np.linalg.det(M) > 0

    def _collect_cluster(self, simplex_id, point):
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
        groups = self._retrieve_groups(points)
        subtris = []
        for gid, subset in groups.items():
            subset = np.asarray(subset)
            if subset.shape[0] < 3:
                continue
            B = self.V_delaunay.find_simplex(subset)
            Pi = HalvingSplitTree(subset)
            buckets = self._local_bucket_partition(subset, B)
            for buck in buckets:
                if len(buck) < 3:
                    continue
                tri = self._subproblem_triangulation(buck, Pi)
                if tri is not None:
                    subtris.append((buck, tri))
        if not subtris:
            return Delaunay(points)
        return self._global_merge(subtris)

    def _retrieve_groups(self, points):
        return {gid: points[idxs] for gid, idxs in self.groups.items()}

    def _local_bucket_partition(self, subset, B):
        buckets = {}
        for i, simplex in enumerate(B):
            if simplex < 0:
                continue
            cluster = self._collect_cluster(simplex, subset[i])
            for t in cluster:
                buckets.setdefault(t, []).append(subset[i])
        return list(buckets.values())

    def _subproblem_triangulation(self, subset_points, Pi):
        pts = np.array(subset_points)
        if len(pts) < 3:
            return None
        order = Pi.preorder()
        order = [i for i in order if i < len(pts)]
        ordered = pts[order]
        return Delaunay(ordered)

    def _global_merge(self, subtris):
        if not subtris:
            return None
        all_pts = np.vstack([b for b, _ in subtris])
        return Delaunay(all_pts)

# --- Performance measurement and plotting ---
ns = [10, 50, 100, 250, 500]
train_times = []
operation_times = []

for n in ns:
    samples = [np.random.rand(n,2) for _ in range(10)]
    sid = SelfImprovingDelaunay()
    t0 = time.perf_counter()
    sid.train(samples)
    t1 = time.perf_counter()
    train_times.append(t1 - t0)

    new_pts = np.random.rand(n,2)
    t2 = time.perf_counter()
    sid.operation(new_pts)
    t3 = time.perf_counter()
    operation_times.append(t3 - t2)

print("Training times:", train_times)
print("Operation times:", operation_times)

# Plot results
plt.figure()
plt.plot(ns, train_times, marker='o', label='Training Phase')
plt.plot(ns, operation_times, marker='s', label='Operation Phase')
plt.xlabel('Number of points (n)')
plt.ylabel('Time (seconds)')
plt.title('Performance: Training vs Operation')
plt.legend()
plt.grid(True)
plt.show()
