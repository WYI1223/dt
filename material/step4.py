# step4.py：轻量化 BST 数组版（节省内存）

import step1, step2, step3

# 树节点采用数组结构：每个节点是 (split_index, left_idx, right_idx)
def build_approximate_bst_array(prob):
    nodes = []  # 最终树节点列表
    stack = [(0, len(prob) - 1, None, None)]  # (left, right, parent_idx, is_left)
    index_map = {}  # 记录每个区间的根在 nodes 中的位置

    while stack:
        left, right, parent_idx, is_left = stack.pop()
        if left > right:
            node_idx = None
        else:
            total = sum(prob[left:right + 1])
            acc = 0
            for i in range(left, right + 1):
                acc += prob[i]
                if acc >= total / 2:
                    root_index = i
                    break
            node_idx = len(nodes)
            nodes.append([root_index, None, None])  # 暂时空的左右孩子
            # 将子任务压栈
            stack.append((root_index + 1, right, node_idx, False))
            stack.append((left, root_index - 1, node_idx, True))

        if parent_idx is not None and node_idx is not None:
            if is_left:
                nodes[parent_idx][1] = node_idx
            else:
                nodes[parent_idx][2] = node_idx

    return nodes  # 返回数组形式的树结构

def build_all_di_trees(prob_matrix):
    di_trees = []
    for i, prob in enumerate(prob_matrix):
        tree_array = build_approximate_bst_array(prob)
        di_trees.append(tree_array)
    return di_trees

if __name__ == "__main__":
    n = 10
    training_data = step1.collect_training_data(n)
    v_list = step2.build_v_list(training_data, v_length=n)
    _, prob_matrix = step3.estimate_distributions(training_data, v_list, n)
    di_trees = build_all_di_trees(prob_matrix)
    print(f"构建了 {len(di_trees)} 棵 Di 树")