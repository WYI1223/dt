# entropy_struct.py

import bisect
import math
from collections import defaultdict

###############################################################################
# 1. EntropyTrieNode：Trie 中的节点类型
###############################################################################

class EntropyTrieNode:
    def __init__(self):
        # children: dict from next symbol (字符或整型标识) 到子节点 EntropyTrieNode
        self.children = {}       # { symbol: EntropyTrieNode, ... }
        # freq_counter: 统计“来到本节点后，各个子字符出现的次数”
        #              用于后续构建分布敏感 BST
        self.freq_counter = defaultdict(int)
        # bst_root: 该节点上构造的分布敏感 BST 的根节点，供 query 时使用
        self.bst_root = None     # type: DistSensitiveBSTNode or None
        # output_label: 如果本节点正好是一条完整输出字符串的末尾，则存储该“输出”本身
        #               否则为 None
        self.output_label = None

    def is_leaf(self):
        # 只判断是否是完整字符串的终点，而非只看 children
        return self.output_label is not None

###############################################################################
# 2. DistSensitiveBSTNode：分布敏感 BST 的节点
###############################################################################

class DistSensitiveBSTNode:
    __slots__ = ("symbol", "left", "right")

    def __init__(self, symbol):
        """
        symbol: 本节点所代表的字符（或整型标识）。left/right 指向子树。
        """
        self.symbol = symbol
        self.left = None
        self.right = None

###############################################################################
# 3. DistSensitiveBST：构造一棵近似最优的加权搜索树（Optimal BST）
#
#    这里使用经典的动态规划 O(n^2) 构建最优二叉搜索树算法（Knuth 优化留待后续）
#    输入 keys = [k1,k2,...,km] （已排序），weights = [w1,w2,...,wm]（概率或频次总和=1）
#    输出一棵 BST，节点存放 symbol = ki，满足查找期望代价最小。
###############################################################################

class DistSensitiveBST:
    def __init__(self, keys, weights):
        """
        keys:    已排序的不重复关键字列表（下一步可能出现的“字符”）
        weights: 对应于 keys 的概率或权重（浮点数），和不一定精确到 1，但相对大小正确。

        构造过程：
         1. 对 keys 及 weights 做一轮动态规划，构造最优 BST。
         2. 存储每个子问题的根节点索引，用于递归建树。
        """

        assert len(keys) == len(weights), "keys 和 weights 长度必须相同"
        self.keys = keys
        self.n = len(keys)
        # 如果只有一个或没有 keys，简单处理
        if self.n == 0:
            self.root = None
            return
        if self.n == 1:
            node = DistSensitiveBSTNode(keys[0])
            self.root = node
            return

        # prefix_sum[i] = sum(weights[:i])，方便快速计算区间和
        prefix_sum = [0.0] * (self.n + 1)
        for i in range(self.n):
            prefix_sum[i+1] = prefix_sum[i] + weights[i]

        # cost[i][j]: 构造 keys[i..j] 最优 BST 的最小加权搜索代价
        # root_choice[i][j]: 构造时在 keys[i..j] 区间下，哪个 k 作为根使代价最小
        cost = [[0.0]*(self.n) for _ in range(self.n)]
        root_choice = [[0]*self.n for _ in range(self.n)]

        # 初始化长度为 1 的子区间
        for i in range(self.n):
            cost[i][i] = weights[i]
            root_choice[i][i] = i

        # 对于长度 L = 2 .. n
        for L in range(2, self.n+1):
            for i in range(0, self.n - L + 1):
                j = i + L - 1
                total_weight = prefix_sum[j+1] - prefix_sum[i]

                # 初始化一个很大值
                min_cost = math.inf
                best_k = i

                # Bruteforce 在 [i..j] 之间枚举根 k
                for k in range(i, j+1):
                    left_cost = cost[i][k-1] if k-1 >= i else 0.0
                    right_cost = cost[k+1][j] if k+1 <= j else 0.0
                    # 总代价 = left_cost + right_cost + total_weight （因为访问根也要加 total_weight）
                    cur_cost = left_cost + right_cost + total_weight
                    if cur_cost < min_cost:
                        min_cost = cur_cost
                        best_k = k

                cost[i][j] = min_cost
                root_choice[i][j] = best_k

        # 通过 root_choice 自顶向下建 BST
        def build_subtree(i, j):
            if i > j:
                return None
            k = root_choice[i][j]
            node = DistSensitiveBSTNode(keys[k])
            node.left  = build_subtree(i, k-1)
            node.right = build_subtree(k+1, j)
            return node

        self.root = build_subtree(0, self.n-1)

    def query(self, symbol):
        """
        在这棵加权 BST 中查找 symbol。如果存在就返回对应的节点，
        否则返回 None。调用者需在拿到节点后继续去下一级的 Trie 节点。
        """
        cur = self.root
        while cur:
            if symbol == cur.symbol:
                return cur
            elif symbol < cur.symbol:
                cur = cur.left
            else:
                cur = cur.right
        return None

###############################################################################
# 4. B_Pi_Structure：整体接口，将训练数据插入 Trie，然后在每个Trie节点
#    上构造 DistSensitiveBST。训练数据形式为列表 of (input_string, output_string) 对。
###############################################################################

class B_Pi_Structure:
    def __init__(self, del_triangulation = None):
        # root: Trie 的根节点
        self.root = EntropyTrieNode()
        self._delV = del_triangulation

    def _insert_output(self, output_string):
        """
        把一条完整的 output_string（元字符序列、如三角形索引串或 Lehmer 码）插入到 Trie 中，
        并沿途更新各节点的 freq_counter（统计出现某字符的次数）。
        """
        node = self.root
        for c in output_string:
            # 在当前节点增计 freq_counter
            node.freq_counter[c] += 1
            # 向子节点推进
            if c not in node.children:
                node.children[c] = EntropyTrieNode()
            node = node.children[c]
        # 到达字符串末尾，标记该节点为叶子，并设置 output_label
        node.output_label = output_string

    def _build_all_bsts(self, node):
        """
        递归遍历 Trie，把 node.freq_counter 中记录的“c→频次”转换成 keys/weights，
        然后构造 node.bst_root = DistSensitiveBST(keys, weights)。
        接着对所有子节点递归执行。
        """
        if node is None:
            return

        # 如果 node.freq_counter 非空，就需要基于它构建 BST
        if node.freq_counter:
            # keys: 将“下一字符的所有可能值”按升序排序
            keys = sorted(node.freq_counter.keys())
            # weights: 按 keys 顺序取对应的次数（或概率）
            weights = [node.freq_counter[c] for c in keys]
            # 注意：这里用 freq（出现次数）构造 BST，等价于用概率比例
            node.bst_root = DistSensitiveBST(keys, weights)

        # 递归对子节点构造
        for child in node.children.values():
            self._build_all_bsts(child)

    def build(self, training_strings):
        """
        training_strings: 一个列表，元素为 (input_string, output_string)。
          - 对于 B 结构：input_string 是 I|_{G′_j} 的某种编码，output_string 是对应三角形索引的序列。
          - 对于 Π 结构：input_string 同理，output_string 是 Lehmer 码或叶子遍历编码。

        过程：
         1. 仅把 output_string 插入 Trie，用以统计各节点的“后续字符”出现频次。
         2. 遍历整棵 Trie，为每个非叶节点构造 node.bst_root。
        """

        # 1) 插入所有 output_string
        for inp, outp in training_strings:
            # 只需要把 outp 插到 Trie（输入 inp 仅用于映射训练输入→训练输出）
            # 若需要同时把 inp 存起来做调试，可改动此处
            self._insert_output(outp)

        # 2) 对整个 Trie 递归构造 BST
        self._build_all_bsts(self.root)

    def _fallback_bruteforce(self, input_string):
        """
        当 query 时发现某一步在 Trie/BST 中无法匹配（也就是训练集中未见过的后续字符路径）时，
        可以退回到暴力计算接口。这里你需要提供一个“暴力计算真正 output_string” 的函数。

        举例：如果这是 B 结构（查询三角形序列），可调用某个暴力点定位算法（如 Walk 算法）
        返回真正的三角形序列。
        """
        # input_string is G'_j, a list of point‐indices.
        # We need to return a list of Face objects from DelV.locate_face.
        if self._delV is None:
            raise RuntimeError("No template triangulation provided for fallback")

        faces = []
        for idx in input_string:
            # you need a mapping from idx→Point;
            # assume you also stored a point‐list in the instance:
            x, y = self._point_coords[idx].x, self._point_coords[idx].y
            face = self._delV.locate_face(x, y)
            faces.append(face)
        return faces
        raise KeyError(f"未在训练集中见过 output 路径 {output_string}，需要暴力回退实现。")

    def query(self, input_string):
        """
        给定 input_string（对应某次查询的输入，比如 I|_{G′_j} 的点编码），
        本方法返回 output_string：在训练集上出现的“最优” output 序列；
        若中途某一步在 BST 中找不到匹配字符，则调用 _fallback_bruteforce。

        期望耗时：O(H + |input_string|)，H 是熵值级别代价，|input_string| 为输入长度。
        """

        node = self.root
        outp = []  # 用来拼接输出字符
        for c in input_string:
            # 第一步：在当前节点的 BST（node.bst_root）里做加权查找
            if node.bst_root is None:
                # 没有训练时记录过任何子字符，需要暴力回退
                return self._fallback_bruteforce(input_string)

            bst_node = node.bst_root.query(c)
            if bst_node is None:
                # 训练集中没出现过“此节点后续为 c” 的情况，需要暴力回退
                return self._fallback_bruteforce(input_string)

            # 找到后缀字符 c，对应下一个 Trie 节点
            outp.append(c)
            node = node.children[c]

        # 整个 input_string 扫描完之后，我们在 Trie 中到了 node
        # 如果 node 就是叶子，直接返回完整 output_label
        if node.is_leaf():
            return node.output_label

        # 如果 node 不是叶子，说明训练集中所有 output 都比 input_string 更长，
        # 需要按 node.output_label（如果存在）或暴力回退
        if node.output_label is not None:
            return node.output_label

        return self._fallback_bruteforce(input_string)
