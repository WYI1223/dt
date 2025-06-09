# operation.py

import random
from collections import defaultdict
from split_tree_build import construct_split_tree, build_sub_split_tree
from delaunay import Triangulation, compute_delaunay, delaunay_to_voronoi, merge_two_delaunay
from geometry import Point


def operation_phase(I_new_coords, state):
    """
    运行阶段主函数。

    参数：
      - I_new_coords: 新输入点集列表，形式 [(x0,y0), (x1,y1), ..., (x_{n−1},y_{n−1})]
                      与训练时相同的全局下标含义。
      - state: 来自 training_phase 的字典，包含：
          * G_prime: 近似分区列表 [G'_0, G'_1, ..., G'_K]
          * V      : 模板点集（Point 列表）
          * DelV   : 模板 Delaunay（三角网 Triangulation 对象）
          * B_struct : dict，其中 B_struct[j] 是“点定位”近熵结构
          * Pi_struct: dict，其中 Pi_struct[j] 是“拆分树”近熵结构

    返回：
      - Del_Inew: 新输入 I_new 的 Delaunay（三角网 Triangulation 对象）
    """

    # 从 state 解包
    Gp = state["G_prime"]
    V = state["V"]
    DelV = state["DelV"]
    B_struct = state["B_struct"]
    Pi_struct = state["Pi_struct"]

    n = len(I_new_coords)  # 假设 n 与训练时一致

    # 1) 按 G_prime 对 I_new_coords 做分组
    #    I_new_coords 是一个长度为 n 的列表，索引对应全局下标 0..n−1
    I_new = [Point(x, y) for (x, y) in I_new_coords]

    # I_G0 对应 G'_0，大小常数，用暴力建 Delaunay
    I_G0 = [I_new[i] for i in Gp[0]]

    # 其它组打平
    I_Gplus = {}
    for j in range(1, len(Gp)):
        I_Gplus[j] = [I_new[i] for i in Gp[j]]

    # 2) 在线查询 B 和 Π，得到每个 j≥1 下标的定位三角形 t0 及子树先序顺序
    # anchor_triangle[i] = t0：表示下标 i 在 DelV 中的锚三角形
    anchor_triangle = {}

    # Q_preorder[j] 是 Gp[j] 中下标按子树先序排列的列表
    Q_preorder = {}

    for j in range(1, len(Gp)):
        if not Gp[j]:
            continue

        # 2a) 构造 input_string_B = Gp[j] 本身（全局下标列表），调用 B_struct[j].query()
        in_str_B = Gp[j].copy()
        out_str_B = B_struct[j].query(in_str_B)
        # out_str_B 应当是一个与 in_str_B 等长的三角形 ID 列表
        # 设 out_str_B[k] 对应 in_str_B[k] 在 DelV 上的锚三角形

        # 记录 anchor_triangle
        for idx, t_idx in zip(in_str_B, out_str_B):
            anchor_triangle[idx] = t_idx

        # 2b) 构造 input_string_Pi = Gp[j] 本身，调用 Pi_struct[j].query()
        in_str_Pi = Gp[j].copy()
        out_str_Pi = Pi_struct[j].query(in_str_Pi)
        # out_str_Pi 是按“子集拆分树先序”返回的一串全局下标
        Q_preorder[j] = out_str_Pi

    # 3) 用 BFS 在 DelV 中对每个 j、每个点 i∈Gp[j] 找冲突三角形 Δ[i]
    from conflict_region import bfs_conflict_region
    Delta = {}  # Delta[i] = 冲突三角形 ID 列表

    for j in range(1, len(Gp)):
        for global_i in Gp[j]:
            t0 = anchor_triangle[global_i]
            p = I_new[global_i]
            # bfs_conflict_region(DelV, t0, Point)
            conflict_tris = bfs_conflict_region(DelV, t0, p)
            Delta[global_i] = conflict_tris

    # 4) 对每个 j、每个 t_idx，将各点 global_i 放到桶 Q_{j,t_idx} 中
    Q_buckets = {j: defaultdict(list) for j in range(1, len(Gp))}
    for j in range(1, len(Gp)):
        for global_i in Gp[j]:
            for t_idx in Delta[global_i]:
                Q_buckets[j][t_idx].append(global_i)

    # 5) 对每个 j、每个 t_idx，按先序顺序对 Q_{j,t_idx} 排序，并剪出对应的拆分子树
    #    SplitT_sub[j][t_idx] = 拆分子树的根节点
    SplitT_sub = {j: {} for j in range(1, len(Gp))}

    for j in range(1, len(Gp)):
        # 先取对应的拆分树（Π）父树根节点
        # 但我们在训练阶段只构造了近熵查询结构，实际的父树并未保存下来。
        # 因此，这里要重新构造“父集拆分树” Π_j；
        #  父集坐标 = [I_new[i] for i in Gp[j]]，本地索引 0..|Gp[j]|-1 对应 全局 Gp[j]
        coords_parent = [I_new[i] for i in Gp[j]]
        parent_indices = list(range(len(Gp[j])))
        parent_root = construct_split_tree(coords_parent, parent_indices, coords_parent)

        # 子集先序 Q_preorder[j] 已给出 “全局下标”顺序
        # 但 build_sub_split_tree 需要“父树先序下标”形式的“本地”索引列表，
        # 所以先把全局 Q_preorder[j] 转为对应的“本地下标”顺序
        global_to_local = {global_i: local_idx for local_idx, global_i in enumerate(Gp[j])}
        Q_local_preorder = [global_to_local[i] for i in Q_preorder[j] if i in global_to_local]

        for t_idx, bucket in Q_buckets[j].items():
            # bucket 是一个 global_i 列表，先转换为本地下标列表，再按 Q_local_preorder 排序
            bucket_local = [global_to_local[i] for i in bucket]
            # 按先序 Q_local_preorder 筛选、排序
            sorted_local = [i for i in Q_local_preorder if i in set(bucket_local)]
            # 用 build_sub_split_tree 得到子拆分树根节点
            sub_root = build_sub_split_tree(parent_root, sorted_local)
            SplitT_sub[j][t_idx] = sub_root

    # 6) 对每个 j、每个 t_idx，基于 SplitT_sub[j][t_idx] 做局部 Delaunay 构造
    Del_sub = {j: {} for j in range(1, len(Gp))}
    for j in range(1, len(Gp)):
        for t_idx, st_root in SplitT_sub[j].items():
            # build_delaunay_from_splittree：按拆分树先序插点得到 Del(Q_{j,t})
            from delaunay_from_splittree import build_delaunay_from_splittree
            # 需要构造 Q_{j,t} 对应的坐标列表传给函数 —— 但 build_delaunay_from_splittree 已经接受“全局坐标”与拆分树：
            Del_sub[j][t_idx] = build_delaunay_from_splittree(st_root, I_new)

    # 7) 合并局部 Delaunay 到模板 Voronoi，再恢复成大 Delaunay
    # 7a) 把 DelV 转成全局 Voronoi
    Vor_global = delaunay_to_voronoi(DelV)

    # 对 DelV 中的每个三角形 ID t_idx，将所有 j ≥ 1 且 t_idx ∈ DelV.triangles 的点集合汇总为 P_t
    # 并且把这些点在各自的局部 Delaunay 里转换为局部 Voronoi，再合并到本地 Voronoi，最后再把本地 Voronoi 与全局 Voronoi 合并。
    for t_idx in range(DelV.num_triangles()):
        # 收集 P_t：所有 j ≥ 1 且 t_idx ∈ Δ[i] 的 i
        P_t = []
        for j in range(1, len(Gp)):
            for global_i in Gp[j]:
                if t_idx in Delta[global_i]:
                    P_t.append(global_i)

        # 收集各个 j 的局部 Voronoi
        Vor_locals = []
        for j in range(1, len(Gp)):
            if t_idx in Del_sub[j]:
                Del_sub_jt = Del_sub[j][t_idx]
                # 把 Del_sub_jt 转到 Voronoi
                Vor_locals.append(delaunay_to_voronoi(Del_sub_jt))

        # 7b) 把这些 Vor_locals 合并成一个 Voronoi 小图 Vor_Pt
        #     这里假设你已有一个函数 merge_voronoi_list(list_of_Voronoi) 可以做这件事
        from delaunay import merge_voronoi_list, geode_merge
        Vor_Pt = merge_voronoi_list(Vor_locals)

        # 7c) 把 Vor_Pt “粘”到全局 Vor_global 的三角形 t_idx 位置，得到新的 Vor_global
        Vor_global = geode_merge(Vor_global, t_idx, Vor_Pt)

    # 7d) 把合并后的全局 Voronoi 转回 Delaunay，得到 Del(V ∪ ⋃_{j≥1} I_new_Gpj)
    Del_VplusG = voronoi_to_delaunay(Vor_global)

    # 8) 从 Del_VplusG 中拆出 “纯 V 三角形” 与 “纯 (⋃_{j≥1} I_new_Gpj) 三角形”
    triangles_V_rest = []
    triangles_Gplus = []
    for tri in Del_VplusG.triangles:
        vs = tri.vertices()
        coords = [(v.x, v.y) for v in vs]
        # 判断 vs 都在 V 里，或都不在 V 里
        in_V = [(abs(v.x - p.x) < 1e-9 and abs(v.y - p.y) < 1e-9) for p in V for v in vs]
        # 简单做法：若三个顶点都属于 V，放入 triangles_V_rest；否则若都不在 V，放入 triangles_Gplus
        if all(any(abs(v.x - p.x) < 1e-9 and abs(v.y - p.y) < 1e-9 for p in V) for v in vs):
            triangles_V_rest.append(tri)
        elif all(not any(abs(v.x - p.x) < 1e-9 and abs(v.y - p.y) < 1e-9 for p in V) for v in vs):
            triangles_Gplus.append(tri)

    # 9) 暴力构造 G'_0 组的小 Delaunay
    Del_G0 = compute_delaunay(I_G0)

    # 10) 合并两部分：Triangulation(I_Gplus) 与 Del_G0
    #     首先把 “I_Gplus 部分的三个顶点与三角形”封成一个 Triangulation 对象
    Tri_Gplus = Triangulation([I_new[vtx] for tri in triangles_Gplus for vtx in tri.vertices()],
                              [(0, 1, 2)])  # placeholder：你需要通过三角形对象把节点索引与下标映射正确
    # 这里假设你已有“从三角形列表构造 Triangulation” 的接口，实际需要把 triangles_Gplus 转为你 Triangulation 的输入格式。
    # 接下来直接调用 merge_two_delaunay
    Del_Inew = merge_two_delaunay(Tri_Gplus, Del_G0)

    return Del_Inew
