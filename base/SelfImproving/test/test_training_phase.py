from base.SelfImproving.training import training_phase
import random

def generate_train_samples(num_samples=3, sample_size=5, max_try=1000):
    """生成不重复的训练样本；同一集合不同顺序也算重复"""
    seen = set()                      # 用哈希集合去重，顺序不敏感
    samples = []
    tries = 0
    while len(samples) < num_samples and tries < max_try:
        pts = tuple(sorted((random.uniform(5, 15), random.uniform(0, 7.5))
                            for _ in range(sample_size)))   # 排序消除顺序差异
        if pts not in seen:
            seen.add(pts)
            samples.append(list(pts))
        tries += 1

    if len(samples) < num_samples:
        raise RuntimeError("无法在给定范围内生成足够的不重复样本，请放宽条件或增大 max_try")

    return samples

# 示例：生成 3 个训练样本，每个样本包含 5 个随机点
train_samples = generate_train_samples(num_samples=3, sample_size=5)
# print(train_samples)
# 打印生成的训练样本
for idx, sample in enumerate(train_samples):
    print(f"Sample {idx}:")
    for pt in sample:
        print(f"  ({pt[0]:.4f}, {pt[1]:.4f})")

state = training_phase(train_samples, n=5)

print("G_prime =", state["G_prime"])
print("模板 V =", [(p.x, p.y) for p in state["V"]])
print("B_struct keys =", list(state["B_struct"].keys()))
print("Pi_struct keys =", list(state["Pi_struct"].keys()))
