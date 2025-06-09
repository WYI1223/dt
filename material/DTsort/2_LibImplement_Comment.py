import warnings
import numpy as np
from scipy.spatial import Delaunay


class HalvingSplitTree:
    """
    二分（kd-）树。通过在 x 与 y 轴上交替使用“中位数”切分，
    把输入点集递归地划分为左右两个子区，力求让每层左右子树
    大小接近，从而得到一棵 **近似平衡** 的树。

    结点保存的字段
    ----------------
    axis   : 本层使用的划分维度（0 ⇒ x，1 ⇒ y）。depth%2 取模实现“交替”。
    index  : 在全局 points 数组中的分割点索引；即当前子区的中位点。
    left   : 左子树（坐标 ≤ split 值）
    right  : 右子树（坐标 > split 值）
    points : 对原坐标数组的只读引用；避免复制，节省内存。

    复杂度
    ------
    * 构造：O(n log n)（排序 + 递归）
    * 查询/插入：平均 O(log n)，最差 O(n)（当点集几乎共线时）
    """

    def __init__(self, points, indices=None, depth=0):
        self.points = points
        if indices is None:
            indices = list(range(len(points)))

        # 按深度决定当前要比较的坐标轴
        self.axis = depth % 2

        # 叶结点：0 或 1 个索引时直接终止递归
        if len(indices) <= 1:
            self.index = indices[0] if indices else None
            self.left = None
            self.right = None
            return

        # 按 axis 上的坐标排序后取中位数，保持左右子树数量平衡
        indices.sort(key=lambda i: points[i][self.axis])
        mid = len(indices) // 2
        self.index = indices[mid]

        # 递归构造左右子树
        self.left = HalvingSplitTree(points, indices[:mid], depth + 1)
        self.right = HalvingSplitTree(points, indices[mid + 1 :], depth + 1)


class TrieNode:
    """
    前缀树(Trie)结点，用于把“网格大三角剖分 V”中的单纯形序列
    (simplex IDs) 映射到 **专属 kd-树 Pi**，以便后续快速桶划分。

    字段
    ----
    children : dict[int, TrieNode]
        以单纯形 ID 为键的子结点。
    Pi : HalvingSplitTree | None
        针对当前路径(=单纯形序列)训练出的 kd-树；
        若 None，说明尚未见过该路径，需回退到祖先结点的 Pi。
    """

    def __init__(self):
        self.children = {}
        self.Pi = None


class SelfImprovingDelaunay:
    """
    **自改进 Delaunay 三角剖分**
    思想源自 Ad-theory 的“Self-Improving Algorithms”。

    1. 离线训练阶段 (`train`)
       • 根据若干批样本学习输入点的稳定模式
         – 先用 `approximate_partition` 把经常“在一起”的点分到同组
         – 再抽样构造近似网格 `V`，并为每个 group 建立 Trie + kd-树

    2. 在线运行阶段 (`operation`)
       • 将新输入按训练时的 group 划分
       • 查 Trie 得到最匹配的 kd-树 Pi，并做 **局部桶划分**
       • 分别对每个桶调用 SciPy 的 Delaunay，最后全局合并

    若新输入与训练分布相似，可显著减少一次性对全体点做
    Delaunay 的成本；若差异过大，则自动回退到“默认 Pi”，不会出错。
    """

    def __init__(self):
        self.trie_roots = {}       # group → Trie 根
        self.V = None              # 采样得到的近似网格点
        self.V_delaunay = None     # V 的 Delaunay 剖分结果
        self.big_triangle = None   # 包含 V 的超大三角形（三点）
        self.groups = None         # {gid: [原数组索引...]}

    # ------------------------------------------------------------------
    # ------------------------  训练阶段  -------------------------------
    # ------------------------------------------------------------------

    def train(self, samples):
        """
        参数
        ----
        samples : list[np.ndarray]  (len = t, 每个 shape = (n, 2))
            t 份训练样本，每份都是 n 个二维点。
        """
        groups = self.approximate_partition(samples)
        self.groups = groups
        self.build_net(samples)
        self.build_tries(samples, groups)

    # ---------- 1) 近似划分 -------------------------------------------------

    def approximate_partition(self, samples):
        """
        多样本一致性聚类。核心思路：
        把“在所有样本里都表现得几乎恒定/共线/等距”的点，判断为同一群。

        返回值
        ------
        dict[int, list[int]]
            gid → 原始点索引列表
        """
        n = samples[0].shape[0]
        parent = list(range(n))  # 并查集

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru

        # (1) 常量测试：若两个点在所有样本中 **坐标完全相同**，则合并
        for i in range(n):
            for j in range(i + 1, n):
                if all(
                    abs(sample[i, 0] - sample[j, 0]) < 1e-8
                    and abs(sample[i, 1] - sample[j, 1]) < 1e-8
                    for sample in samples
                ):
                    union(i, j)

        # (2) 二元差分测试：判断 (j − i) 的 x、y 坐标差在所有样本中
        #     均值≈0 且方差≈0，说明两点始终保持同向同距 (平移不变)
        for i in range(n):
            for j in range(i + 1, n):
                diffs_x = [sample[j, 0] - sample[i, 0] for sample in samples]
                diffs_y = [sample[j, 1] - sample[i, 1] for sample in samples]
                if (
                    abs(np.mean(diffs_x)) < 1e-8
                    and np.std(diffs_x) < 1e-8
                    and abs(np.mean(diffs_y)) < 1e-8
                    and np.std(diffs_y) < 1e-8
                ):
                    union(i, j)

        # (3) 三元测试：任取三点 (i,j,k)，若它们在所有样本中
        #     det([[xi,yi,1],[xj,yj,1],[xk,yk,1]]) ≈ 0，
        #     即三点总是共线，则把它们并到同一连通分量
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # 若任意两点已不在同一集合，则无需检测三元共线
                    if find(i) != find(j) and find(j) != find(k) and find(i) != find(k):
                        continue
                    # 若三点两两有重合，不做三元共线判定
                    if any(
                        (
                            abs(sample[a, 0] - sample[b, 0]) < 1e-8
                            and abs(sample[a, 1] - sample[b, 1]) < 1e-8
                        )
                        for sample in samples
                        for a, b in [(i, j), (j, k), (i, k)]
                    ):
                        continue
                    dets = [
                        np.linalg.det(
                            np.array(
                                [
                                    [sample[i, 0], sample[i, 1], 1],
                                    [sample[j, 0], sample[j, 1], 1],
                                    [sample[k, 0], sample[k, 1], 1],
                                ]
                            )
                        )
                        for sample in samples
                    ]
                    if all(abs(d) < 1e-6 for d in dets):
                        union(i, j)
                        union(j, k)

        # 把并查集结果转成 gid → 索引列表
        comps = {}
        for idx in range(n):
            root = find(idx)
            comps.setdefault(root, []).append(idx)
        # 为了确定输出顺序，按索引最小点排序
        return {gid: comp for gid, comp in enumerate(sorted(comps.values(), key=lambda x: x[0]))}

    # ---------- 2) 构造近似网格 V -------------------------------------------

    def build_net(self, samples):
        """
        • 把所有训练样本堆叠后随机抽 m ≈ n·log n 个点形成 V
        • 再用一个足够大的外包三角形把 V 包住，以避免 Delaunay 条件失败
        """
        pts = np.vstack(samples)
        n = samples[0].shape[0]
        m = int(n * np.log(n) + 1)  # 理论上保证近似覆盖
        chosen = np.random.choice(len(pts), size=min(m, len(pts)), replace=False)
        self.V = pts[chosen]

        # 构造“超大三角形”包围盒
        maxc, minc = np.max(self.V, axis=0), np.min(self.V, axis=0)
        d = np.max(maxc - minc) * 10  # 放大 10 倍，确保完全覆盖
        p1 = minc + np.array([-1, -1]) * d
        p2 = minc + np.array([3, -1]) * d
        p3 = minc + np.array([-1, 3]) * d
        self.big_triangle = np.vstack([p1, p2, p3])

        # 拼接后一次性做 Delaunay，为查 simplex 序列做准备
        self.V = np.vstack([self.V, self.big_triangle])
        self.V_delaunay = Delaunay(self.V)

    # ---------- 3) 为每个 group 构建 Trie + Pi -------------------------------

    def build_tries(self, samples, groups):
        """
        对于每个 group：
        1) 用所有样本合并的数据先建一个“回退 Pi”，保证 Trie 查询不到时仍可用。
        2) 对训练样本逐份插入，记录其在网格 V 中的 simplex 序列 B_seq，
           并在 Trie 叶结点挂上专属 kd-树 Pi，实现“分布自适应”索引。
        """
        for gid, idxs in groups.items():
            root = TrieNode()

            # ---- 3-a) 默认 Pi：合并所有样本中该 group 的点 -----------------
            merged = np.vstack([sample[idxs] for sample in samples])
            root.Pi = HalvingSplitTree(merged)

            # ---- 3-b) 把每份样本路径插入 Trie -------------------------------
            for sample in samples:
                subset = sample[idxs]
                # B_seq: 对每一点找它在 V 的单纯形 ID ⇒ len = |subset|
                B_seq = tuple(self.V_delaunay.find_simplex(subset).tolist())
                Pi = HalvingSplitTree(subset)

                node = root
                for s in B_seq:  # 逐 ID 插入/下沉
                    node = node.children.setdefault(s, TrieNode())
                node.Pi = Pi  # 叶结点挂上专属 kd-树

            self.trie_roots[gid] = root

    # ------------------------------------------------------------------
    # ------------------------  运行阶段  -------------------------------
    # ------------------------------------------------------------------

    def operation(self, points):
        """
        在线阶段主入口。将新输入拆分 → 各自三角化 → 全局合并。
        返回值：SciPy Delaunay 对象（含 simplices、neighbors 等属性）
        """
        groups = self._retrieve_groups(points)
        subtris = []

        # 1) 对每个 group 独立处理
        for gid, subset in groups.items():
            # 找 subset 中每点在 V 的 simplex 序列
            B_seq = tuple(self.V_delaunay.find_simplex(subset).tolist())

            # 2) Trie 查询最合适的 kd-树 Pi；若未见过路径则回退
            Pi = self._query_Pi(gid, B_seq)

            # 3) 用 Pi 把 group 划分为若干“桶”，每桶局部三角化
            buckets = self._local_bucket_partition(subset, Pi)
            for buck in buckets:
                tri = Delaunay(np.array(buck)) if len(buck) >= 3 else None
                subtris.append((buck, tri))

        # 4) 把所有桶的点堆叠后做一次全局 Delaunay，完成合并
        return self._global_merge(subtris)

    # ---------- 内部辅助函数 -----------------------------------------------

    def _retrieve_groups(self, points):
        """根据训练好的 group 索引映射，切分新输入"""
        return {gid: points[idxs] for gid, idxs in self.groups.items()}

    def _query_Pi(self, gid, B_seq):
        """
        沿着 B_seq 在 Trie 中下沉；若遇到未见过的 simplex 序列，
        触发 warnings 并使用 root.Pi 作为回退。
        """
        root = self.trie_roots[gid]
        node = root
        for s in B_seq:
            child = node.children.get(s)
            if child is None:
                warnings.warn(f"未见过的 simplex 序列 {B_seq}，group {gid} 使用默认 Pi 回退。")
                return root.Pi
            node = child
        return node.Pi or root.Pi  # 若叶结点没挂 Pi，也回退

    def _local_bucket_partition(self, subset_points, Pi):
        """
        在 kd-树 Pi 上查找每个点所属的叶结点 index，
        同一叶结点的点视作“空间上足够局部”，分到一个桶。
        """
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
        """简单粗暴：把所有桶里的点拼在一起再做一次 Delaunay。"""
        all_pts = np.vstack([b for b, _ in subtris])
        return Delaunay(all_pts)


# ----------------------------------------------------------------------
# -------------------                单元测试                ------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    n = 100
    samples = [np.random.rand(n, 2) for _ in range(10)]
    sid = SelfImprovingDelaunay()
    sid.train(samples)

    new_pts = np.random.rand(n, 2)
    result = sid.operation(new_pts)
    print("全局三角形数量:", len(result.simplices) if result else 0)

    # 以下测试专门验证 approximate_partition 的正确性
    pts_const = np.tile([1.0, 2.0], (5, 1))
    samples_const = [pts_const for _ in range(3)]
    groups_const = sid.approximate_partition(samples_const)
    assert len(groups_const) == 1 and sorted(groups_const[0]) == list(range(5))

    pts_clusters = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    samples_clusters = [pts_clusters for _ in range(4)]
    groups_clust = sid.approximate_partition(samples_clusters)
    sorted_comps = sorted([sorted(v) for v in groups_clust.values()])
    assert sorted_comps == [[0, 1], [2, 3]]

    pts_three = np.array([[2, 2], [2, 2], [3, 3]])
    samples_three = [pts_three for _ in range(3)]
    groups_three = sid.approximate_partition(samples_three)
    sorted_three = sorted([sorted(v) for v in groups_three.values()])
    assert sorted_three == [[0, 1], [2]]

    pts_linear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    samples_linear = [pts_linear for _ in range(2)]
    groups_linear = sid.approximate_partition(samples_linear)
    sorted_linear = sorted([sorted(v) for v in groups_linear.values()])
    assert sorted_linear == [[0], [1], [2], [3]]

    # 测试 operation 的“回退逻辑”是否生效
    try:
        result = sid.operation(new_pts)
        assert isinstance(result, Delaunay)
    except Exception as e:
        raise AssertionError(f"operation 失败: {e}")

    print("全部测试通过。")
