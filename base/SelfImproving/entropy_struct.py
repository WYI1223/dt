from collections import defaultdict
import math
from base.SelfImproving.split_tree_build import construct_split_tree
from base.DCEL.dcel import DCEL as Triangulation
from base.DCEL.face import Face

# -----------------------------------------------------------------------------
# 1. EntropyTrieNode: 节点类型
# -----------------------------------------------------------------------------
class EntropyTrieNode:
    def __init__(self):
        # 子节点映射：symbol -> EntropyTrieNode
        self.children = {}
        # 该节点处各后续符号出现频次 (symbol -> count)
        self.freq_counter = defaultdict(int)
        # 分布敏感 BST 根，用于快速下一符号查找
        self.bst_root = None
        # 如果此节点正好是某条输出的末尾，保存完整输出序列
        self.output_label = None

    def is_leaf(self):
        return self.output_label is not None

# -----------------------------------------------------------------------------
# 2. DistSensitiveBSTNode: 分布敏感 BST 的单节点
# -----------------------------------------------------------------------------
class DistSensitiveBSTNode:
    __slots__ = ("symbol", "left", "right")

    def __init__(self, symbol):
        self.symbol = symbol
        self.left = None
        self.right = None

# -----------------------------------------------------------------------------
# 3. DistSensitiveBST: 最优加权二叉搜索树的动态规划构造
# -----------------------------------------------------------------------------
class DistSensitiveBST:
    def __init__(self, keys, weights):
        assert len(keys) == len(weights), "keys 与 weights 长度必须相同"
        self.keys = keys
        n = len(keys)
        # 边界情况
        if n == 0:
            self.root = None
            return
        if n == 1:
            self.root = DistSensitiveBSTNode(keys[0])
            return

        # 前缀加权和
        prefix_sum = [0.0] * (n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + weights[i]

        # cost[i][j] = 构造 keys[i..j] 的最小搜索代价
        cost = [[0.0]*n for _ in range(n)]
        # root_choice[i][j] = 构造 keys[i..j] 时的最优根索引
        root_choice = [[0]*n for _ in range(n)]

        for i in range(n):
            cost[i][i] = weights[i]
            root_choice[i][i] = i

        # 子区间长度 L
        for L in range(2, n+1):
            for i in range(0, n-L+1):
                j = i + L - 1
                total_w = prefix_sum[j+1] - prefix_sum[i]
                best_cost = math.inf
                best_k = i
                for k in range(i, j+1):
                    left_cost  = cost[i][k-1] if k-1 >= i else 0.0
                    right_cost = cost[k+1][j] if k+1 <= j else 0.0
                    c = left_cost + right_cost + total_w
                    if c < best_cost:
                        best_cost = c
                        best_k = k
                cost[i][j] = best_cost
                root_choice[i][j] = best_k

        # 递归构造 BST
        def build_sub(i, j):
            if i > j:
                return None
            k = root_choice[i][j]
            node = DistSensitiveBSTNode(keys[k])
            node.left  = build_sub(i, k-1)
            node.right = build_sub(k+1, j)
            return node

        self.root = build_sub(0, n-1)

    def query(self, symbol):
        # 常规 BST 查找
        cur = self.root
        while cur:
            if symbol == cur.symbol:
                return cur
            if symbol < cur.symbol:
                cur = cur.left
            else:
                cur = cur.right
        return None

# -----------------------------------------------------------------------------
# 4. B_Pi_Structure: 统一的近熵查询结构
# -----------------------------------------------------------------------------
class B_Pi_Structure:
    def __init__(self, del_triangulation: Triangulation=None, point_coords=None):
        """
        del_triangulation: 模板 Delaunay Triangulation，用于 B 结构的回退
        point_coords    : 全局点坐标列表，用于回退时查找坐标
        """
        self.root          = EntropyTrieNode()
        self._delV         = del_triangulation
        self._point_coords = point_coords

    def _insert_output(self, output_string):
        # 将完整的 output_string 序列插入 Trie，并统计 freq
        node = self.root
        for c in output_string:
            node.freq_counter[c] += 1
            if c not in node.children:
                node.children[c] = EntropyTrieNode()
            node = node.children[c]
        node.output_label = output_string

    def _build_all_bsts(self, node: EntropyTrieNode):
        # 递归为每个 Trie 节点构造分布敏感 BST
        if node.freq_counter:
            keys = sorted(node.freq_counter.keys())
            weights = [node.freq_counter[k] for k in keys]
            node.bst_root = DistSensitiveBST(keys, weights)
        for child in node.children.values():
            self._build_all_bsts(child)

    def build(self, training_strings):
        """
        training_strings: List of (input_string, output_string) pairs.
        只用 output_string 构建 Trie 和 BST。
        """
        for inp, outp in training_strings:
            self._insert_output(outp)
        self._build_all_bsts(self.root)

    def _fallback_bruteforce(self, input_string):
        # B 结构回退：如果有 del_triangulation，则按点定位暴力回退
        if self._delV is not None:
            if self._point_coords is None:
                raise RuntimeError("需要提供 point_coords 才能回退点定位")
            faces = []
            for idx in input_string:
                p = self._point_coords[idx]
                face = self._delV.locate_face(p.x, p.y)
                faces.append(face)
            return faces
        # Π 结构回退：重建 split tree，再先序
        if self._point_coords is None:
            raise RuntimeError("需要提供 point_coords 才能回退拆分树先序")
        coords = [ self._point_coords[i] for i in input_string ]
        local_idx = list(range(len(input_string)))
        root = construct_split_tree(coords, local_idx, coords)
        result = []
        def dfs(node):
            if node.is_leaf():
                result.append(input_string[node.indices[0]])
            else:
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        return result

    def query(self, input_string):
        """
        查询给定的 input_string（List of symbols），返回完整 output_string。
        如果训练时未见过某路径，则回退到暴力实现。
        """
        node = self.root
        outp = []
        for c in input_string:
            if node.bst_root is None:
                return self._fallback_bruteforce(input_string)
            bst_node = node.bst_root.query(c)
            if bst_node is None:
                return self._fallback_bruteforce(input_string)
            outp.append(bst_node.symbol)
            node = node.children.get(bst_node.symbol)
            if node is None:
                return self._fallback_bruteforce(input_string)
        # 到达 Trie 末端
        if node.is_leaf():
            return node.output_label
        if node.output_label is not None:
            return node.output_label
        # 其余情况回退
        return self._fallback_bruteforce(input_string)
