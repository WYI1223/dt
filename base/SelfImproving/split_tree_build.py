from base.SelfImproving.split_tree import SplitTreeNode
# split_tree_build.py

def construct_split_tree(points, point_indices, point_coords):
    """
    递归构造 halving split tree。
    参数：
      points         :点 全局点坐标数组（Point 列表）
      point_indices  : 当前子集的全局下标列表
      point_coords   : 全局坐标（方便索引，例如 points[i].x/y）
    返回：
      root : SplitTreeNode 根节点
    """
    # 如果子集只有一个点，创建叶子
    if len(point_indices) == 1:
        return SplitTreeNode(point_indices)

    # 1) 找当前子集的最小 bounding rectangle
    xs = [point_coords[i].x for i in point_indices]
    ys = [point_coords[i].y for i in point_indices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width  = max_x - min_x
    height = max_y - min_y

    # 2) 在更大的正方形上选择最长边所在方向做切分
    node = SplitTreeNode(point_indices)
    if width >= height:
        # 竖直切
        mid = (min_x + max_x)/2
        node.is_vertical = True
        node.cut_coord = mid
        left_indices  = [i for i in point_indices if point_coords[i].x <= mid]
        right_indices = [i for i in point_indices if point_coords[i].x >  mid]
    else:
        # 水平切
        mid = (min_y + max_y)/2
        node.is_vertical = False
        node.cut_coord = mid
        left_indices  = [i for i in point_indices if point_coords[i].y <= mid]
        right_indices = [i for i in point_indices if point_coords[i].y >  mid]

    # 由于是 halving split tree，保证两侧都至少为 1
    node.left  = construct_split_tree(points, left_indices,  point_coords)
    node.left.parent = node
    node.right = construct_split_tree(points, right_indices, point_coords)
    node.right.parent = node
    return node


def build_sub_split_tree(parent_root, Q_sorted):
    """
    parent_root : 父 SplitT 的根节点
    Q_sorted    : 按父树先序排列好的子集 global index 列表
    返回：子集 Q 的拆分树根节点
    """
    # 先把 Q_sorted 做成集合，方便快速 membership check
    Q_set = set(Q_sorted)

    # 1) 建立 global_index -> 父树叶子节点 的映射
    leaf_map = {}  # { global_index: parent_leaf_node }
    def map_leaves(node):
        if node is None:
            return
        if node.is_leaf():
            idx = node.indices[0]  # 叶子的全局索引
            # 只有当该索引属于父树叶子集合（即所有叶子都是有效的）
            leaf_map[idx] = node
        else:
            map_leaves(node.left)
            map_leaves(node.right)
    map_leaves(parent_root)

    # 2) 子树根的占位引用（真正的根会在首次插入时创建）
    sub_root = None

    # 3) 记录哪些 父节点 已复制到 “子树” 中：copied_nodes
    #    key: 父节点引用；value: 对应子树中新节点的引用
    copied_nodes = {}

    # 4) 按 Q_sorted 顺序逐个插入
    for i in Q_sorted:
        # 找到父树中 i 对应的叶子节点
        parent_leaf = leaf_map[i]

        # a) 从叶子向上找，直到遇到已复制节点 或 到达父树根
        path_stack = []
        cur = parent_leaf
        while (cur not in copied_nodes) and (cur is not parent_root):
            path_stack.append(cur)
            cur = cur.parent

        # b) 现在 cur 要么是已复制节点，要么是父树根
        if cur in copied_nodes:
            sub_parent = copied_nodes[cur]
        else:
            # 父树根尚未复制：把父树根复制到子树根
            # 复制时只保留属于 Q_set 的索引
            filtered = [idx for idx in cur.indices if idx in Q_set]
            sub_parent = SplitTreeNode(filtered, parent=None)
            sub_parent.is_vertical = cur.is_vertical
            sub_parent.cut_coord   = cur.cut_coord
            sub_root = sub_parent
            copied_nodes[cur] = sub_parent

        # c) 反向从 path_stack 创建子节点，一路“下钻”到叶子
        #    path_stack 中存储的是从 parent_leaf 向上到 cur（不含 cur）之间的节点
        while path_stack:
            parent_node = path_stack.pop()  # 父树里的节点

            # 复制 parent_node 时，只保留属于 Q_set 的索引
            filtered = [idx for idx in parent_node.indices if idx in Q_set]
            sub_node = SplitTreeNode(filtered, parent=sub_parent)
            sub_node.is_vertical = parent_node.is_vertical
            sub_node.cut_coord   = parent_node.cut_coord

            # 判断 parent_node 在父树中是其父节点的左或右孩子
            # 如果 parent_node.parent 是 None（意味着 parent_node 就是父树根），
            # 由于上面已经处理了 cur is parent_root 的情形，这里一般不会发生。
            if parent_node.parent and (parent_node is parent_node.parent.left):
                sub_parent.left = sub_node
            else:
                sub_parent.right = sub_node

            # 记录映射关系
            copied_nodes[parent_node] = sub_node
            # 继续向下
            sub_parent = sub_node

        # 最后 parent_leaf（本身是叶子）对应的 sub_node 已在 copied_nodes 中
        # 这时 sub_parent 已经指向这一叶子节点

    return sub_root