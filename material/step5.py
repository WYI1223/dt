import step1, step2, step3, step4

# 组大小
GROUP_SIZE = 1  # 如果你有分组的话改为实际值；否则为1（每位独立）

def locate_bucket(x, i, di_trees, v_list):
    """
    使用数组结构的 Di 树查找元素 x 应该落入的区间。
    每棵树是一个 list，每个节点是 [split_index, left_idx, right_idx]。
    """

    mapped_index = i // GROUP_SIZE  # 如果不分组，这里就是 i
    tree = di_trees[mapped_index]

    idx = 0  # 从根节点（第0个节点）开始
    while idx is not None:
        split_index, left_idx, right_idx = tree[idx]
        if x < v_list[split_index]:
            idx = left_idx
        else:
            idx = right_idx
    return split_index

def bucket_classify(new_data, di_trees, v_list):
    """
    输入：
    - new_data: 新的测试数据
    - di_trees: 构建好的轻量数组树
    - v_list: 带边界哨兵的 V-list

    输出：
    - buckets: 大小为 n+1 的桶列表
    """
    n = len(new_data)
    buckets = [[] for _ in range(n + 1)]

    for i, x in enumerate(new_data):
        k = locate_bucket(x, i, di_trees, v_list)
        k = max(0, min(k, n))  # 防止越界
        buckets[k].append(x)

    return buckets

if __name__ == "__main__":
    n = 10
    training_data = step1.collect_training_data(n)
    v_list = step2.build_v_list(training_data, v_length=n)
    _, prob_matrix = step3.estimate_distributions(training_data, v_list, n)
    di_trees = step4.build_all_di_trees(prob_matrix)

    new_input = step1.generate_input(n)
    print("新输入序列：", new_input)

    buckets = bucket_classify(new_input, di_trees, v_list)
    for idx, bucket in enumerate(buckets):
        print(f"桶 {idx}: {bucket}")
