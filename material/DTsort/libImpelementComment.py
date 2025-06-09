import time  # 导入时间模块，用于测量算法执行和训练时间
import numpy as np  # 导入 NumPy，用于高效地处理数组和数值计算
from scipy.spatial import Delaunay  # 从 SciPy 库中导入 Delaunay，用于执行三角剖分


class HalvingSplitTree:
    """
    半分拆分树（类似于 KD 树的一种变体），用于对二维点集进行递归划分。
    每一层递归交替使用 x 轴和 y 轴进行分割，直到每个子集只剩下 1 个或 0 个点。
    """
    def __init__(self, points, indices=None, depth=0):
        # points: 原始点集，shape=(n, 2)
        # indices: 当前子树包含的点在 points 中的索引列表
        # depth: 当前递归深度，用于决定分割轴（depth%2，即 0 表示 x 轴，1 表示 y 轴）
        if indices is None:
            # 初始调用时，若未提供索引列表，则默认包含所有点
            indices = list(range(len(points)))
        self.axis = depth % 2  # 分割轴：偶数层按 x，奇数层按 y

        # 递归终止条件：如果子集大小 <=1，则不再分割
        if len(indices) <= 1:
            self.index = indices[0] if indices else None  # 唯一点的索引或空
            self.left = None  # 左子树
            self.right = None  # 右子树
            return

        # 按当前轴对索引对应的点坐标进行排序
        indices.sort(key=lambda i: points[i][self.axis])
        mid = len(indices) // 2  # 中点位置
        self.index = indices[mid]  # 当前节点选择中位点

        # 构建左右子树，depth+1 以切换分割轴
        self.left = HalvingSplitTree(points, indices[:mid], depth + 1)
        self.right = HalvingSplitTree(points, indices[mid+1:], depth + 1)

    def preorder(self):
        """
        前序遍历：按 "根 -> 左 -> 右" 的顺序返回所有节点索引。
        用于生成树结构在内存中的扁平表示。
        """
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
    基础字典树节点，用于保存每个组对应的三角剖分信息。
    children: 子节点字典，键为点索引，值为下一级 TrieNode。
    B: 存储每个样本在网格中所在的 Delaunay 单纯形索引列表。
    Pi: 存储当前子集对应的 HalvingSplitTree。
    """
    def __init__(self):
        self.children = {}  # 子节点映射
        self.B = None  # Delaunay 单纯形编号序列
        self.Pi = None  # 对应的 HalvingSplitTree


class SelfImprovingDelaunay:
    """
    自适应三角剖分算法：通过训练样本预构建加速结构，
    在后续查询时利用已学习的信息提升三角剖分性能。
    """
    def __init__(self):
        self.trie_roots = {}    # 每个组对应的 Trie 字典树根节点
        self.V = None           # 从样本中随机抽取的点集，用于构建参考网格
        self.V_delaunay = None  # 基于 V 构建的 Delaunay 对象
        self.big_triangle = None  # 包含所有 V 的超大三角形三个顶点

    def train(self, samples):
        """
        训练接口：给定多个二维点数组样本，
        1. 近似划分样本点的等价类（approximate_partition），
        2. 构建全局参考网格（build_net），
        3. 为每个组构建 Trie 加速结构（build_tries）。
        samples: 列表，每个元素是 shape=(n,2) 的 NumPy 数组。
        """
        groups = self.approximate_partition(samples)
        self.build_net(samples)
        self.build_tries(samples, groups)

    def approximate_partition(self, samples):
        """
        对样本的点索引进行近似分组，
        将在所有样本中相对位置和几何关系一致的点归为一组。
        1. 对完全重合的点合并；
        2. 对平均偏移量和方差均极小的点对合并；
        3. 对共线三元组进行处理，将组成同一直线的小三元集合并；
        最终返回字典：组ID -> 索引列表。
        """
        n = samples[0].shape[0]  # 每个样本的点数
        parent = list(range(n))  # 并查集初始化

        def find(u):
            # 路径压缩查找
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            # 合并两集合
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru

        # 遍历所有点对，若在所有样本中坐标完全相等，则合并
        for i in range(n):
            for j in range(i+1, n):
                if all(abs(sample[i,0] - sample[j,0]) < 1e-8 and abs(sample[i,1] - sample[j,1]) < 1e-8 for sample in samples):
                    union(i, j)
        # 若在所有样本中偏移量平均值和标准差都趋近于零，也合并
        for i in range(n):
            for j in range(i+1, n):
                diffs_x = [sample[j,0] - sample[i,0] for sample in samples]
                diffs_y = [sample[j,1] - sample[i,1] for sample in samples]
                if (abs(np.mean(diffs_x)) < 1e-8 and np.std(diffs_x) < 1e-8 and
                    abs(np.mean(diffs_y)) < 1e-8 and np.std(diffs_y) < 1e-8):
                    union(i, j)
        # 对每一组三元组，若在某样本中三点共线，可判断进一步合并
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if find(i) != find(j) and find(j) != find(k) and find(i) != find(k):
                        continue  # 不在同一初步集合中则跳过
                    # 如果任意两点在所有样本中重合，则跳过
                    if any(
                        (abs(sample[i,0]-sample[j,0])<1e-8 and abs(sample[i,1]-sample[j,1])<1e-8) or
                        (abs(sample[j,0]-sample[k,0])<1e-8 and abs(sample[j,1]-sample[k,1])<1e-8) or
                        (abs(sample[i,0]-sample[k,0])<1e-8 and abs(sample[i,1]-sample[k,1])<1e-8)
                        for sample in samples
                    ):
                        continue
                    # 计算行列式判断三点是否共线
                    dets = [
                        np.linalg.det(np.array([
                            [sample[i,0], sample[i,1], 1],
                            [sample[j,0], sample[j,1], 1],
                            [sample[k,0], sample[k,1], 1]
                        ]))
                        for sample in samples
                    ]
                    # 如果所有 dets 都接近 0，则认为三点共线，合并它们
                    if all(abs(d) < 1e-6 for d in dets):
                        union(i, j)
                        union(j, k)

        # 构造结果字典：根索引 -> 成员列表
        comps = {}
        for idx in range(n):
            root = find(idx)
            comps.setdefault(root, []).append(idx)
        # 按根索引排序并重新编号组 ID
        return {gid: comp for gid, comp in enumerate(sorted(comps.values(), key=lambda x: x[0]))}

    def build_net(self, samples):
        """
        构建全局参考网格：
        1. 将所有样本的点堆叠，随机抽取 O(n log n) 个点作为 V；
        2. 在 V 外扩一个足够大的超三角形，确保后续任何查询点都在其内部；
        3. 对扩充后的点集执行 Delaunay 三角剖分，得到 V_delaunay。
        """
        pts = np.vstack(samples)  # 将所有样本拼接成一个大点集
        n = samples[0].shape[0]
        m = int(n * np.log(n) + 1)  # 随机抽取目标数 O(n log n)
        chosen = np.random.choice(len(pts), size=min(m, len(pts)), replace=False)
        self.V = pts[chosen]
        maxc, minc = np.max(self.V, axis=0), np.min(self.V, axis=0)
        d = max(maxc - minc) * 10  # 超三角形边长，10 倍范围
        # 构造三个顶点，形成一个足够大的三角形
        p1 = minc + np.array([-1, -1]) * d
        p2 = minc + np.array([3, -1]) * d
        p3 = minc + np.array([-1, 3]) * d
        self.big_triangle = np.vstack([p1, p2, p3])
        # 将超三角形顶点并入 V
        self.V = np.vstack([self.V, self.big_triangle])
        # 对扩充后的 V 构建 Delaunay
        self.V_delaunay = Delaunay(self.V)

    def build_tries(self, samples, groups):
        """
        对每个组，遍历所有样本，构建一棵 Trie 树：
        - 树路径由组中点的索引序列决定；
        - 叶节点存储该子集在参考网格中的 simplex 索引 B，以及对应的分割树 Pi。
        """
        for gid, idxs in groups.items():
            root = TrieNode()
            # 针对每个训练样本，沿路径插入
            for sample in samples:
                key = tuple(sorted(idxs))  # 保证路径索引有序
                node = root
                for k in key:
                    node = node.children.setdefault(k, TrieNode())
                # 样本在 V_delaunay 中对应的单纯形索引列表
                node.B = self._compute_B(sample[idxs])
                # 基于该子集构建局部分割树
                node.Pi = self._compute_Pi(sample[idxs])
            self.trie_roots[gid] = root

    def _compute_B(self, points):
        """
        根据参考网格 V_delaunay，查找每个点所在的 simplex 索引
        返回列表，与输入点顺序一一对应。
        """
        return self.V_delaunay.find_simplex(points).tolist()

    def _compute_Pi(self, points):
        """
        对给定的点子集构建 HalvingSplitTree，作为局部分割结构 Pi。
        """
        return HalvingSplitTree(points)

    def operation(self, points):
        """
        针对新的查询点集 points：
        1. 检索落入哪个预训练组（_retrieve_groups）；
        2. 对每组分别进行局部子问题：计算 B 和 Pi，并基于 Pi 对点进行划分和三角剖分；
        3. 合并所有子剖分结果，返回整体的 Delaunay 结构。
        """
        groups = self._retrieve_groups(points)
        subtris = []
        for gid, subset in groups.items():
            B = self._compute_B(subset)
            Pi = self._compute_Pi(subset)
            for p in subset:
                # 对单点作局部剖分示例
                tri = self._subproblem_triangulation([p], Pi)
                subtris.append((p.reshape(1,2), tri))
        return self._global_merge(subtris)

    def _retrieve_groups(self, points):
        # TODO: 实现真正的分组检索，当前仅简单返回所有点为一组
        return {0: points}

    def _local_bucket_partition(self, points, B):
        # TODO: 基于 B 信息，将 points 划分到不同子桶，目前每点独立一桶
        return [[p] for p in points]

    def _subproblem_triangulation(self, subset_points, Pi):
        # 对长度 >=3 的子集执行 Delaunay，否则返回 None
        return Delaunay(np.array(subset_points)) if len(subset_points) >= 3 else None

    def _global_merge(self, subtris):
        # 将所有子剖分的点拼接，执行全局 Delaunay
        all_pts = np.vstack([b for b, t in subtris])
        return Delaunay(all_pts)


# 性能测量部分
ns = [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
sample_count = 10
train_times = {}
exec_times = {}

for n in ns:
    samples = [np.random.rand(n, 2) for _ in range(sample_count)]
    sid = SelfImprovingDelaunay()
    if n <= 100:
        t0 = time.perf_counter()
        sid.train(samples)  # 训练时间测量
        train_times[n] = time.perf_counter() - t0
    else:
        train_times[n] = None
        sid.build_net(samples)  # 大样本仅构建参考网格
    new_pts = np.random.rand(n, 2)
    t0 = time.perf_counter()
    sid.operation(new_pts)  # 执行时间测量
    exec_times[n] = time.perf_counter() - t0

# 基于 O(n^3) 模型估计大样本训练时间
t_ref = train_times[100]
est_train_times = {n: (train_times[n] if train_times[n] is not None else t_ref * (n / 100) ** 3) for n in ns}

import matplotlib.pyplot as plt

plt.figure()
plt.plot(ns, [est_train_times[n] for n in ns], label='Training')  # 绘制训练时间曲线
plt.plot(ns, [exec_times[n] for n in ns], label='Execution')  # 绘制执行时间曲线
plt.xlabel('Number of points (n)')
plt.ylabel('Time (seconds)')
plt.title('Performance of SelfImprovingDelaunay')  # 标题
plt.legend()
plt.show()

# 打印各 n 值的训练和执行时间统计
for n in ns:
    tt = est_train_times[n]
    te = exec_times[n]
    if train_times[n] is None:
        print(f"n={n}: Training time ≈ {tt:.4f}s (estimated), Execution time = {te:.4f}s")
    else:
        print(f"n={n}: Training time = {tt:.4f}s, Execution time = {te:.4f}s")
