# training.py
import math
import random
from base.SelfImproving.split_tree_build import construct_split_tree
from base.SelfImproving.Delaunay_face_routing_simplified import compute_delaunay
from base.SelfImproving.learn_partition import learn_approx_partition   # k-means 版 learn_approx_partition
from base.SelfImproving.entropy_struct import B_Pi_Structure
from base.DCEL.vertex import Vertex as Point

def extract_template_points(train_samples, size):
    """
    从所有训练样本里收集所有坐标后，去重，再随机抽 size 个作为 V。
    """
    # 1) 用 set 去重
    unique_coords = set()
    for I in train_samples:
        for (x, y) in I:
            # 如果坐标本身是浮点数，建议先 round 到某个精度后再去重
            coord = (round(x, 9), round(y, 9))
            unique_coords.add(coord)

    # 2) 如果去重后总数少于 size，直接全拿；否则随机抽 size 个
    all_unique = list(unique_coords)
    if len(all_unique) <= size:
        chosen = all_unique
    else:
        chosen = random.sample(all_unique, size)

    # 3) 把它们封成 Point 对象
    V = [Point(x, y) for (x, y) in chosen]
    return V


def collect_training_strings_for_B(train_samples, DelV, Gpj):
    """
    为“点定位 B”收集训练对 (input_string, output_string)：
      - input_string: Gpj 列表本身（若要更严谨可以编码坐标，但 ID 列表即可）
      - output_string: Gpj 中每个点在模板 DelV 上的三角形索引列表
    需要 DelV.locate_triangle_for_point(p: Point) 能返回三角形 ID。
    """
    training_pairs = []
    for I in train_samples:
        # I: [(x,y), ...]，长度 = n
        # 1) 提取 I_j 的坐标列表
        I_j = [I[i] for i in Gpj]
        # 2) input_string 直接使用下标 Gpj
        in_str = Gpj.copy()
        # 3) 对 I_j 中每个点做点定位，得到 t_idx 序列
        out_str = []
        for p_xy in I_j:
            p = Point(p_xy[0], p_xy[1])
            t_idx = DelV.faces.index(DelV.locate_face(p.x, p.y))
            out_str.append(t_idx)
        training_pairs.append((in_str, out_str))
    return training_pairs

def collect_training_strings_for_Pi(train_samples, Gpj):
    """
    为“拆分树 Π”收集训练对 (input_string, output_string)：
      - input_string: Gpj 列表本身
      - output_string: 将 I_j 构造成拆分树后，输出其“叶子先序对应的全局下标”序列
    """
    training_pairs = []
    for I in train_samples:
        # 1) 提取 I_j 坐标 & 构造 coords_j
        coords_j = [Point(I[i][0], I[i][1]) for i in Gpj]
        # 2) input_string 使用 Gpj
        in_str = Gpj.copy()
        # 3) 构造拆分树，并生成先序叶子对应的全局下标
        root_sub = construct_split_tree(coords_j, list(range(len(Gpj))), coords_j)

        out_str = []
        def preorder_map(node):
            if node is None:
                return
            if node.is_leaf():
                local_idx = node.indices[0]          # 叶子在 coords_j 中的下标
                global_idx = Gpj[local_idx]          # 对应原 I 的全局下标
                out_str.append(global_idx)
            else:
                preorder_map(node.left)
                preorder_map(node.right)

        preorder_map(root_sub)
        training_pairs.append((in_str, out_str))

    return training_pairs

def training_phase(train_samples, n, k = 10):
    """
    训练阶段：
      - train_samples: 训练样本列表，每个样本是一个长度为 n 的点集 I = [(x,y), ...]
      - n: 每个 I 的点数
    返回一个字典 state，包括：
      * G_prime   : 近似分区 G' = [G'_0, G'_1, ...]
      * V         : 模板点集（Point 列表，长度 = n）
      * DelV      : 模板 Delaunay (Triangulation 对象)
      * B_struct  : dict, B_struct[j] = B_Pi_Structure() 用于点定位 B 查询
      * Pi_struct : dict, Pi_struct[j] = B_Pi_Structure() 用于拆分树 Π 查询
    """

    # ----------------------------------------------------------------------------
    # 1) 用 k-means 版 learn_approx_partition 得到 G_prime
    # ----------------------------------------------------------------------------
    Gp = learn_approx_partition(train_samples,k)
    # Gp[0] 是空列表或常数小列表，Gp[j], j>=1 是下标列表

    # ----------------------------------------------------------------------------
    # 2) 构造模板点集 V，并计算其 Delaunay DelV
    # ----------------------------------------------------------------------------
    V = extract_template_points(train_samples, size=n)
    DelV = compute_delaunay(V)

    # ----------------------------------------------------------------------------
    # 3) 遍历每个组 G'_j (j >= 1)，构造对应的 B / Π 近熵查询结构
    # ----------------------------------------------------------------------------
    B_struct = {}
    Pi_struct = {}
    for j, Gpj in enumerate(Gp):
        # 跳过 G'_0
        if j == 0:
            continue

        # 3.1) 收集 B 训练对
        training_pairs_B = collect_training_strings_for_B(train_samples, DelV, Gpj)
        bps = B_Pi_Structure()
        bps.build(training_pairs_B)
        B_struct[j] = bps

        # 3.2) 收集 Π 训练对
        training_pairs_Pi = collect_training_strings_for_Pi(train_samples, Gpj)
        pis = B_Pi_Structure()
        pis.build(training_pairs_Pi)
        Pi_struct[j] = pis

    # ----------------------------------------------------------------------------
    # 4) 返回 state
    # ----------------------------------------------------------------------------
    state = {
        "G_prime": Gp,
        "V": V,
        "DelV": DelV,
        "B_struct": B_struct,
        "Pi_struct": Pi_struct
    }
    return state
