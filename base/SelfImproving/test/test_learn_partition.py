import math, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def learn_approx_partition(train_samples):
    """
    用高斯混合模型对每个下标 i 构造特征，再做近似分区。

    输入：
      train_samples: 训练样本列表，每个样本是一个点集
                     I=[(x0,y0),(x1,y1),...,(x_{n-1},y_{n-1})]
                     假定所有样本大小都相同 = n。
    输出：
      Gp: 一个列表 [G'_0, G'_1, ..., G'_K]：
        - G'_0 = []（保留，不使用）
        - 对 j=1..K, G'_j 是那些被 GaussianMixture 分到第 j−1 类的下标 i 列表。
    """
    if not train_samples:
        return []

    n = len(train_samples[0])
    m = len(train_samples)

    # 1) 计算每个 i 的平均坐标
    avg_coords = []
    for i in range(n):
        sx = sum(train_samples[s][i][0] for s in range(m))
        sy = sum(train_samples[s][i][1] for s in range(m))
        avg_coords.append((sx/m, sy/m))

    # 2) 确定簇数 K ≈ √n
    # K = max(1, int(math.sqrt(n)))
    K = 3

    # 3) 用 GaussianMixture 做聚类
    X = np.array(avg_coords)
    gmm = GaussianMixture(
        n_components=K,
        covariance_type='full',
        n_init=5,
        max_iter=200,
        random_state=0
    )
    labels = gmm.fit_predict(X)  # 每个 i 的簇索引 0..K-1

    # 4) 构造输出 Gp
    Gp = [[] for _ in range(K+1)]
    for i, c in enumerate(labels):
        Gp[c+1].append(i)
    return Gp


if __name__ == "__main__":
    # —— 示例数据 ——
    m, n = 5, 1000
    k = 3  # 假设有 3 个簇
    layout = "x_split"
    from base.SelfImproving.PointDistribution2 import generate_clustered_points_superfast
    train_samples = [
        generate_clustered_points_superfast(
            n, k,
            x_range=(5.01, 15),
            y_range=(0.01, 7.5),
            std_dev=0.5,
            layout=layout
        )
        for _ in range(m)
    ]
    # 先算 avg_coords，用于绘图
    avg_coords = []
    for i in range(n):
        sx = sum(train_samples[s][i][0] for s in range(m))
        sy = sum(train_samples[s][i][1] for s in range(m))
        avg_coords.append((sx/m, sy/m))

    # 调用分区函数
    Gp = learn_approx_partition(train_samples)

    # 绘图：不同簇用不同颜色
    num_groups = len(Gp) - 1  # 跳过 Gp[0]
    cmap = plt.get_cmap("tab10")  # 最多 10 色

    plt.figure(figsize=(6,6))
    for j in range(1, len(Gp)):
        xs = [avg_coords[i][0] for i in Gp[j]]
        ys = [avg_coords[i][1] for i in Gp[j]]
        plt.scatter(xs, ys,
                    label=f"Group {j}",
                    color=cmap((j-1) % 10),
                    s=30, alpha=0.8)

    plt.legend()
    plt.xlabel("Average X")
    plt.ylabel("Average Y")
    plt.title("Partition by Gaussian Mixture")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

