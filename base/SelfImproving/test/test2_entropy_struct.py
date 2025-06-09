from base.SelfImproving.entropy_struct import B_Pi_Structure

# =============================
# 1. Training Phase
# =============================
def train_entropy_struct(training_data):
    """
    构建熵敏结构的训练函数。

    参数:
    training_data (list of tuples): 每个tuple 为 (input_string, output_string)
    返回:
    B_Pi_Structure 实例
    """
    # 1) 创建结构
    struct = B_Pi_Structure()
    # 2) 构建（只利用 output_string 统计分布）
    struct.build(training_data)
    return struct

# =============================
# 2. Operation (查询) Phase
# =============================
def query_entropy_struct(entropy_struct, input_str):
    """
    使用训练好的熵敏结构进行查询。

    参数:
    entropy_struct (B_Pi_Structure): 已训练好的结构实例
    input_str (str): 查询的输入字符串
    返回:
    输出字符串或 None
    """
    result = entropy_struct.query(input_str)
    return result

# 查询示例
if __name__ == "__main__":
    # 示例训练数据
    training_examples = [
        ("a", "apple"),
        ("ap", "apricot"),
        ("b", "banana"),
        ("ba", "berry"),
        ("c", "cherry"),
    ]

    entropy_struct = train_entropy_struct(training_examples)
    print("训练完成：熵敏结构已构建。")

    # 假设 entropy_struct 已通过训练加载或构造
    tests = ["a", "ap", "ba", "c", "d"]
    for t in tests:
        out = query_entropy_struct(entropy_struct, t)
        print(f"Input: '{t}' -> Output: {out}")
