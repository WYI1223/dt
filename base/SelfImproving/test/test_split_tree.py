# tests/test_split_tree.py

import random
import unittest

# 把你的模块路径替换成真实路径，比如：
from ..split_tree import SplitTreeNode
from ..split_tree_build import construct_split_tree, build_sub_split_tree
from ...DCEL.vertex import Vertex
class TestSplitTree(unittest.TestCase):

    def setUp(self):
        # 生成 n 个随机点
        self.n = 20
        random.seed(123)
        self.points = []
        for _ in range(self.n):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            self.points.append(Vertex(x, y))
        # 全局坐标列表
        self.point_coords = self.points

        # 全局索引 [0,1,2,...,n-1]
        self.full_indices = list(range(self.n))

        # 构造父树
        self.root_full = construct_split_tree(
            self.points, self.full_indices, self.point_coords
        )

    def test_full_tree_structure(self):
        """
        测试构造的 halving split tree 是否符合预期：
        1. 叶子数 == n
        2. 每个节点的 indices 等于其所有子叶 indices 的并集
        3. 切分方向和坐标是否合理
        """
        # 1) 统计叶子节点
        leaves = []
        def collect_leaves(node):
            if node is None:
                return
            if node.is_leaf():
                leaves.append(node)
            else:
                collect_leaves(node.left)
                collect_leaves(node.right)

        collect_leaves(self.root_full)
        self.assertEqual(len(leaves), self.n,
                         msg=f"叶子数应该等于 n={self.n}，实际 {len(leaves)}")

        # 2) 检查每个节点的 indices 与其所有子叶的 indices 并集一致
        def check_indices(node):
            if node is None:
                return
            if node.is_leaf():
                # 叶子节点 indices 长度为 1
                self.assertEqual(len(node.indices), 1)
                self.assertIn(node.indices[0], self.full_indices)
            else:
                # 递归收集所有下层叶子 indices
                child_leaves = []
                def collect_subleaves(nd):
                    if nd is None:
                        return
                    if nd.is_leaf():
                        child_leaves.append(nd.indices[0])
                    else:
                        collect_subleaves(nd.left)
                        collect_subleaves(nd.right)
                collect_subleaves(node)
                # 比较并集
                self.assertEqual(set(node.indices), set(child_leaves),
                                 msg=f"节点 indices {node.indices} 与子叶并集 {child_leaves} 不一致")
                check_indices(node.left)
                check_indices(node.right)

        check_indices(self.root_full)

        # 3) 验证切分方向与坐标是否合理
        def check_split_valid(node):
            if node is None or node.is_leaf():
                return
            # 当前节点的全部点坐标范围
            xs = [self.point_coords[i].x for i in node.indices]
            ys = [self.point_coords[i].y for i in node.indices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            is_v, cut = node.split_info()
            if is_v:
                # 竖直切：cut 必须在 [min_x, max_x]
                self.assertTrue(min_x <= cut <= max_x,
                                msg=f"竖直切: cut={cut} 不在 x 范围 [{min_x}, {max_x}]")
                # 左子集 x <= cut，右子集 x > cut
                for i in node.left.indices:
                    self.assertTrue(self.point_coords[i].x <= cut + 1e-9)
                for i in node.right.indices:
                    self.assertTrue(self.point_coords[i].x >  cut - 1e-9)
            else:
                # 水平切：cut 必须在 [min_y, max_y]
                self.assertTrue(min_y <= cut <= max_y,
                                msg=f"水平切: cut={cut} 不在 y 范围 [{min_y}, {max_y}]")
                # 左子集 y <= cut，右子集 y > cut
                for i in node.left.indices:
                    self.assertTrue(self.point_coords[i].y <= cut + 1e-9)
                for i in node.right.indices:
                    self.assertTrue(self.point_coords[i].y >  cut - 1e-9)

            check_split_valid(node.left)
            check_split_valid(node.right)

        check_split_valid(self.root_full)

    def test_sub_split_tree_on_random_subset(self):
        """
        随机抽取父树叶子索引的一部分 Q，按父树先序生成 Q_sorted，
        再调用 build_sub_split_tree 构造子树，然后校验子树性质：
        1. 子树叶子数 == |Q|
        2. 子树中所有节点的 indices 都属于 Q
        3. 所有内部节点的切分仍然满足 halving 子集的约束
        """
        # 1) 随机选一个子集 Q_indices
        subset_size = random.randint(1, self.n - 1)
        Q_indices = random.sample(self.full_indices, subset_size)

        # 2) 得到父树的完整先序列
        full_preorder = []
        def preorder_collect(node):
            if node is None:
                return
            if node.is_leaf():
                full_preorder.append(node.indices[0])
            else:
                preorder_collect(node.left)
                preorder_collect(node.right)
        preorder_collect(self.root_full)

        # Q_sorted: 在 full_preorder 顺序中筛出 Q_indices
        Q_sorted = [i for i in full_preorder if i in set(Q_indices)]
        # 确保我们真的筛出了正确数量
        self.assertEqual(len(Q_sorted), subset_size,
                         msg=f"Q_sorted 长度 {len(Q_sorted)} 应等于 subset_size {subset_size}")

        # 3) 构造子树
        root_sub = build_sub_split_tree(self.root_full, Q_sorted)

        # 4a) 检查子树的叶子数量
        sub_leaves = []
        def collect_sub_leaves(nd):
            if nd is None:
                return
            if nd.is_leaf():
                sub_leaves.append(nd.indices[0])
            else:
                collect_sub_leaves(nd.left)
                collect_sub_leaves(nd.right)

        collect_sub_leaves(root_sub)
        self.assertEqual(len(sub_leaves), subset_size,
                         msg=f"子树叶子数 {len(sub_leaves)} 应等于 subset_size {subset_size}")
        # 并且叶子下标集应该恰等于 Q_indices（集合相等即可，无需顺序一致）
        self.assertEqual(set(sub_leaves), set(Q_indices),
                         msg=f"子树叶子 indices {sub_leaves} 应与 Q_indices {Q_indices} 集合相等")

        # 4b) 检查子树内部节点的 indices 都属于 Q_indices，并且切分合法
        def check_subtree(nd):
            if nd is None:
                return
            # 节点的所有 indices 必须都是 Q_indices 的子集
            for idx in nd.indices:
                self.assertIn(idx, Q_indices,
                              msg=f"子树节点中发现 index {idx} 不在 Q_indices {Q_indices} 中")
            # 如果不是叶子，还要检查它的切分是否合理
            if not nd.is_leaf():
                is_v, cut = nd.split_info()
                left_idx  = nd.left.indices  if nd.left  else []
                right_idx = nd.right.indices if nd.right else []

                # 只要非空，就检查对应切分条件
                if is_v:
                    # 对左侧所有点，x <= cut；对右侧所有点，x > cut
                    for i in left_idx:
                        self.assertTrue(self.point_coords[i].x <= cut + 1e-9,
                                        msg=f"子树竖直切: 左侧点 {i} 的 x={self.point_coords[i].x} 应 <= cut={cut}")
                    for i in right_idx:
                        self.assertTrue(self.point_coords[i].x >  cut - 1e-9,
                                        msg=f"子树竖直切: 右侧点 {i} 的 x={self.point_coords[i].x} 应 > cut={cut}")
                else:
                    # 水平切: 左侧 y <= cut；右侧 y > cut
                    for i in left_idx:
                        self.assertTrue(self.point_coords[i].y <= cut + 1e-9,
                                        msg=f"子树水平切: 左侧点 {i} 的 y={self.point_coords[i].y} 应 <= cut={cut}")
                    for i in right_idx:
                        self.assertTrue(self.point_coords[i].y >  cut - 1e-9,
                                        msg=f"子树水平切: 右侧点 {i} 的 y={self.point_coords[i].y} 应 > cut={cut}")

                # 继续递归检查
                check_subtree(nd.left)
                check_subtree(nd.right)

        check_subtree(root_sub)


if __name__ == "__main__":
    unittest.main()