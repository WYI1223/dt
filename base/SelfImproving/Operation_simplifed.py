from collections import defaultdict

from base.SelfImproving.Delaunay_face_routing_simplified import compute_delaunay
from base.SelfImproving.split_tree_build import construct_split_tree, build_sub_split_tree
from base.SelfImproving.delaunay_from_splittree import build_delaunay_from_splittree
from base.DCEL.vertex import Vertex as Point
from base.DCEL.face import Face
from base.SelfImproving.Delaunay_face_routing_simplified import DelaunayWithRouting as Triangulation


def bfs_conflict_region(triang: Triangulation, start_face, p: Point):
    """
    在 triang 中，从 start_face 出发，BFS 找出所有与点 p 的外接圆冲突的三角形。
    支持 start_face 是 Face 对象或其整数索引。
    使用 Lawson-flip 证书 isDelunayTrain(he, p) 来判断是否冲突。
    """
    from collections import deque
    # 若 start_face 是索引，转换为 Face 对象
    if not isinstance(start_face, Face):
        try:
            start_face = triang.faces[start_face]
        except Exception:
            raise RuntimeError(f"Invalid face key: {start_face}")
    visited = set()
    queue = deque([start_face])
    conflict = []
    while queue:
        f = queue.popleft()
        if f in visited:
            continue
        visited.add(f)
        conflict.append(f)
        # 遍历 f 的所有半边，依证书扩散
        for he in triang.enumerate_half_edges(f):
            if triang.isDelunayTrain(he, p):
                nf = he.twin.incident_face
                if not isinstance(nf, Face):
                    # incident_face 也可能是索引
                    nf = triang.faces[nf]
                if nf not in visited:
                    queue.append(nf)
    return conflict


def operation_phase(I_new_coords, state):
    """
    运行阶段：仅收集“拆分树先序”插入序列。

    输入：
      I_new_coords : List[(x,y)] 新输入点集
      state        : training_phase 返回的字典，包含
                     'G_prime', 'V', 'DelV', 'B_struct', 'Pi_struct'
    返回：
      insertion_seq: List[Point]，按照局部拆分树先序排列的新点列表
                     （每个点出现一次）
    """

    # 解包训练阶段产物
    Gp = state['G_prime']  # 近似分区列表
    V = state['V']  # 模板点集
    DelV = state['DelV']  # 模板 Delaunay
    Pi_struct = state['Pi_struct']  # 拆分树先序的熵结构

    # 构造 Point 对象列表
    I_new = [Point(x, y) for x, y in I_new_coords]

    # 更新所有 Π 结构的点坐标引用，用于回退（若需要）
    for pis in Pi_struct.values():
        pis._point_coords = I_new

    # 按组顺序收集插入序列
    insertion_seq = []
    for j in range(1, len(Gp)):
        if j not in Pi_struct:
            continue
        indices = Gp[j]  # 全局下标列表
        # 得到该组拆分树先序的全局下标序列
        try:
            preorder = Pi_struct[j].query(indices)
        except Exception as e:
            # 回退到暴力重建拆分树
            # print("Error querying Pi_struct for group", j, ":", e)
            coords_j = [I_new[i] for i in indices]
            local_idxs = list(range(len(indices)))
            root = construct_split_tree(coords_j, local_idxs, coords_j)
            # 先序遍历叶子映射回全局下标
            preorder = []

            def dfs(node):
                if node.is_leaf():
                    preorder.append(indices[node.indices[0]])
                else:
                    dfs(node.left)
                    dfs(node.right)

            dfs(root)

        # 按先序把对应的 Point 对象加入结果
        for idx in preorder:
            insertion_seq.append(I_new[idx])

    return insertion_seq


# def operation_phase(I_new_coords, state):
#     """
#     运行阶段（前六步）...
#     返回：按子树插入顺序排列的所有新点列表。
#     """
#     # 解包训练状态
#     Gp        = state['G_prime']
#     V         = state['V']
#     DelV      = state['DelV']
#     B_struct  = state['B_struct']
#     Pi_struct = state['Pi_struct']
#
#     # 全局点对象列表
#     I_new = [Point(x, y) for x, y in I_new_coords]
#
#     # 更新熵结构的运行时点坐标引用
#     for bps in B_struct.values():
#         bps._point_coords = I_new
#     for pis in Pi_struct.values():
#         pis._point_coords = I_new
#
#     # 2) 直接用 Π 结构给出的先序来收集插入序列
#     insertion_seq = []
#     for j in range(1, len(Gp)):
#         if j not in Pi_struct:
#             continue
#         # query 返回的就是该组 Gp[j] 在拆分树叶子先序下的全局下标列表
#         preorder = Pi_struct[j].query(Gp[j])
#         for idx in preorder:
#             insertion_seq.append(I_new[idx])
#
#     # 将按子树先序插入的新点顺序存入列表
#     insertion_seq = []
#     inserted_set = set()
#
#     # 跳过 G0，Gplus 为其余组的全局下标映射
#     I_Gplus = {j: [I_new[i] for i in Gp[j]] for j in range(1, len(Gp))}
#
#     # 1) 熵查询：锚三角形 & 子树先序
#     anchors = {}       # i -> Face
#     Q_preorder = {}    # j -> 全局下标先序列表
#     for j in range(1, len(Gp)):
#         if j not in B_struct or j not in Pi_struct:
#             continue
#         idxs = Gp[j]
#         faces = B_struct[j].query(idxs)
#         for i, f in zip(idxs, faces): anchors[i] = f
#         Q_preorder[j] = Pi_struct[j].query(idxs)
#
#     # 2) 通过 B_struct 的 anchor 三角形作为唯一桶键，不再区分全部冲突面
#     # anchors: i -> Face 已由第1步构建
#     # 直接按 anchors 分桶
#     Q_buckets = {j: defaultdict(list) for j in range(1, len(Gp))}
#     for j in range(1, len(Gp)):
#         if j not in Q_preorder:
#             continue
#         for i in Gp[j]:
#             f = anchors[i]
#             Q_buckets[j][f].append(i)
#
#     # 3) 子树剪枝 & 收集插入顺序
#     for j in range(1, len(Gp)):
#         coords_j = [I_new[i] for i in Gp[j]]
#         parent_root = construct_split_tree(coords_j, list(range(len(Gp[j]))), coords_j)
#         g2l = {g:i for i,g in enumerate(Gp[j])}
#         local_preorder = [g2l[g] for g in Q_preorder.get(j, []) if g in g2l]
#         for f, bucket in Q_buckets[j].items():
#             # 只使用 anchor 三角形对应的 bucket
#             bucket_local = [g2l[i] for i in bucket]
#             order_local  = [x for x in local_preorder if x in set(bucket_local)]
#             # 按先序把点加入插入序列
#             for local_idx in order_local:
#                 global_i = Gp[j][local_idx]
#                 insertion_seq.append(I_new[global_i])
#             # 可选：构造局部 Delaunay
#             sub_root = build_sub_split_tree(parent_root, order_local)
#             _ = build_delaunay_from_splittree(sub_root, I_new)
#
#     return insertion_seq

if __name__ == "__main__":
    # 示例运行：从 training_phase 直接拿 state
    from training import training_phase
    import random
    # 随机生成训练和运行样本
    random.seed(42)
    # 生成 m=10 个训练样本，每个大小 n=20
    m, n = 5, 50
    # train_samples = [
    #     [(random.uniform(5, 15), random.uniform(0, 7.5)) for _ in range(n)]
    #     for __ in range(m)
    # ]
    from PointDistribution2 import generate_clustered_points_superfast
    k=3
    train_samples = [
        generate_clustered_points_superfast(n, k,
                                          x_range=(5, 15),
                                          y_range=(0, 7.5),
                                          std_dev=0.4) for _ in range(n)
        for __ in range(m)
    ]
    state = training_phase(train_samples, n)
    # 生成一个新的测试实例
    # I_new = [(random.uniform(5, 15), random.uniform(0, 7.5)) for _ in range(n)]
    I_new = generate_clustered_points_superfast(n, k,
                                          x_range=(5, 15),
                                          y_range=(0, 7.5),
                                          std_dev=0.4)

    I_vertices = [Point(x, y) for x, y in I_new]
    results = operation_phase(I_new, state)
    delv = state['DelV']
    # delv.draw()
    print(len(results))
    print(len(I_new))

    import time
    start = time.perf_counter()
    delaunay = compute_delaunay(results)
    end = time.perf_counter()
    print("generator",end - start)

    start = time.perf_counter()
    input_delaunay = compute_delaunay(I_vertices)
    end = time.perf_counter()
    print("input_order",end - start)

    from base.DelunayClassic import compute_delaunay as compute_delaunay2
    start = time.perf_counter()
    input_delaunay = compute_delaunay2(I_vertices)
    end = time.perf_counter()
    print("Classic_D", end - start)

    # delaunay.draw()
    # input_delaunay.draw()