# training.py

import json
import random
from learn_partition import learn_approx_partition
from split_tree_build import construct_split_tree
from base.SelfImproving.Delaunay_face_routing_simplified import compute_delaunay
from entropy_struct import B_Pi_Structure
from base.DCEL.vertex import Vertex as Point

def extract_template_points(train_samples, size):
    """
    从所有训练样本中去重后随机抽 size 个点作为模板 V。
    """
    unique = set()
    for I in train_samples:
        for x,y in I:
            # 四舍五入到 9 位防止浮点误差
            unique.add((round(x,9), round(y,9)))
    pts = list(unique)
    if len(pts) <= size:
        chosen = pts
    else:
        chosen = random.sample(pts, size)
    # 返回 Point 对象列表
    return [Point(x,y) for x,y in chosen]

def collect_training_strings_for_B(train_samples, DelV, Gpj):
    """
    收集 B 结构的训练对 (in_str, out_str)，
    in_str = Gpj 列表，out_str = 对应的 Face 列表。
    """
    pairs = []
    for I in train_samples:
        in_str = Gpj.copy()
        out_str = []
        for idx in Gpj:
            x,y = I[idx]
            face = DelV.locate_face(x, y)
            out_str.append(face)
        pairs.append((in_str, out_str))
    return pairs

def collect_training_strings_for_Pi(train_samples, Gpj):
    """
    收集 Π 结构的训练对 (in_str, out_str)，
    out_str = Gpj 对应拆分树先序下标列表。
    """
    pairs = []
    for I in train_samples:
        in_str = Gpj.copy()
        coords = [Point(*I[idx]) for idx in Gpj]
        local = list(range(len(Gpj)))
        root = construct_split_tree(coords, local, coords)
        out_str = []
        def dfs(node):
            if node.is_leaf():
                out_str.append(Gpj[node.indices[0]])
            else:
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        pairs.append((in_str, out_str))
    return pairs

def training_phase(train_samples, n):
    """
    训练阶段，返回一个 state 字典。
    """
    # 1) 近似分区
    G_prime = learn_approx_partition(train_samples)

    # 2) 模板 V 及其 Delaunay
    V = extract_template_points(train_samples, n)
    DelV = compute_delaunay(V)

    # 3) 为每个组 j>=1 构建 B 和 Π
    B_struct  = {}
    Pi_struct = {}
    for j, Gpj in enumerate(G_prime):
        if j == 0 or not Gpj:
            continue

        # B 结构
        train_B = collect_training_strings_for_B(train_samples, DelV, Gpj)
        bps = B_Pi_Structure(del_triangulation=DelV, point_coords=V)
        bps.build(train_B)
        B_struct[j] = bps

        # Π 结构
        train_Pi = collect_training_strings_for_Pi(train_samples, Gpj)
        pis = B_Pi_Structure(point_coords=V)
        pis.build(train_Pi)
        Pi_struct[j] = pis

    return {
        "G_prime":   G_prime,
        "V":         V,
        "DelV":      DelV,
        "B_struct":  B_struct,
        "Pi_struct": Pi_struct
    }

if __name__ == "__main__":
    # —— 示例：用随机数据测试一下 ——
    random.seed(42)
    # 生成 m=10 个训练样本，每个大小 n=20
    m, n = 10, 20
    train_samples = [
        [(random.uniform(5, 15), random.uniform(0, 7.5)) for _ in range(n)]
        for __ in range(m)
    ]

    state = training_phase(train_samples, n)

    # 打印近似分区和模板点集
    Gp = state["G_prime"]
    V  = state["V"]
    Delv = state["DelV"]
    Delv.draw()
    print("G_prime =", Gp)
    print("Template V (coordinates):")
    for p in V:
        print(f"  ({p.x:.6f}, {p.y:.6f})")

    # 将 G_prime 和 V 保存到 JSON，方便后续载入
    out = {
        "G_prime": Gp,
        "V":       [[p.x, p.y] for p in V]
    }
    with open("training_state.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved training_state.json with G_prime and V.")
