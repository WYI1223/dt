# split_tree.py

class SplitTreeNode:
    def __init__(self, point_indices, parent=None):
        """
        point_indices : 这一节点代表的子集（global index 列表）
        parent        : 父节点引用
        """
        self.indices = point_indices  # e.g. [i1, i2, ...] 下标列表
        self.parent  = parent
        self.left    = None
        self.right   = None
        self.is_vertical = True  # True: 竖直切，False: 水平切
        self.cut_coord   = None  # 切分线的 x 或 y 坐标
        # 叶节点时，self.indices 长度为 1，或通过 is_leaf() 判断

    def is_leaf(self):
        return len(self.indices) == 1

    def split_info(self):
        # 返回 (is_vertical, cut_coord)
        return (self.is_vertical, self.cut_coord)
