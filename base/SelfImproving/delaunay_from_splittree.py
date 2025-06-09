# 文件：delaunay_from_splittree.py

from base.SelfImproving.split_tree import SplitTreeNode
from base.SelfImproving.Delaunay_face_routing_simplified import compute_delaunay
from base.Trainglation.TrainClassic import TrainClassic as Triangulation
from base.DCEL.vertex import Vertex as Point
import random

def extract_indices_from_split_tree(node):
    """
    从拆分树 node 中递归提取所有叶子对应的 global index 列表。
    """
    if node is None:
        return []
    if node.is_leaf():
        return [node.indices[0]]
    return (extract_indices_from_split_tree(node.left) +
            extract_indices_from_split_tree(node.right))

def split_tree_preorder(node, order_list):
    """
    把拆分树做先序遍历，将叶子(global index)逐一追加到 order_list 中。
    非叶子节点并不往 order_list 写，只在递归时决定先访问哪边。
    """
    if node is None:
        return
    if node.is_leaf():
        # 直接把 leaf 的全局索引写进“插入顺序”
        order_list.append(node.indices[0])
    else:
        split_tree_preorder(node.left, order_list)
        split_tree_preorder(node.right, order_list)

def build_delaunay_from_splittree(splitT_root, global_points):
    """
    直接利用 insert_point_with_certificate 接口，
    按拆分树先序顺序，把 Q 中的所有点依次插入空 Triangulation。

    - splitT_root  : 对应 Q 集合的拆分树根节点
    - global_points: 全局坐标列表（列表索引即“global index”）
    返回一个 Triangulation 对象，代表 Del(Q)。
    """
    # 1) 按先序遍历拆分树，做一个插入顺序列表
    insertion_order = []
    split_tree_preorder(splitT_root, insertion_order)
    # insertion_order 现在就是 Q_indices 的顺序，长度 m = |Q|

    # 2) 构造一个空的 Triangulation
    #    假设你的 Triangulation 类里有一个单独的 constructor 可以创建空网格
    #    并且提供 insert_point_with_certificate(Point) 方法
    D = Triangulation()  # 空的 Delaunay 三角网

    # 3) 依次把 insertion_order 中对应的点插入 Triangulation
    points = []
    for idx in insertion_order:
        points.append(global_points[idx])
    D = compute_delaunay(points)
    # for idx in insertion_order:
    #     p = global_points[idx]
    #     print(p)
    #     print(D)
    #     D.insert_point_with_certificate(p)
    #     D.draw()
        # 上述方法需要 Triangulation 内部维护“对已有点的 certificate/最近邻”或者
        # “从最后一次插入的三角形开始向外搜冲突区域”的逻辑。
        # 只要保证 insert_point_with_certificate 能在 O(1)~O(log n) 期望内完成，
        # 整体过程就是 O(m) 期望。

    # 4) 返回最终 Triangulation 对象
    return D
