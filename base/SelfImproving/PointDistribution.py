import random
import numpy as np
from scipy.stats import truncnorm


def generate_clustered_points_trunc(n, k,
                                    x_range=(5, 15),
                                    y_range=(0, 7.5),
                                    std_dev=0.5,
                                    weights=None):
    """
    在指定矩形区域内生成 n 个聚类点，点分布为每个簇中心的截断正态。

    参数
    ----
    n : int
        总共生成的点数
    k : int
        簇数
    x_range : tuple(float, float)
        x 轴最小、最大值 (xmin, xmax)
    y_range : tuple(float, float)
        y 轴最小、最大值 (ymin, ymax)
    std_dev : float
        每个簇的标准差
    weights : list[float] or None
        各簇生成点的占比（长度 k，和为1）。若为 None 则均匀分配。

    返回
    ----
    points : list[(float, float)]
        生成的点列表
    """
    # 1) 随机生成 k 个簇中心
    centers = [
        (random.uniform(x_range[0], x_range[1]),
         random.uniform(y_range[0], y_range[1]))
        for _ in range(k)
    ]

    # 2) 确定每个点属于哪个簇
    if weights is None:
        weights = [1.0 / k] * k
    labels = np.random.choice(np.arange(k), size=n, p=weights)

    # 3) 为每个点沿 x、y 维度做截断正态采样
    points = []
    for cid in labels:
        cx, cy = centers[cid]
        # 计算截断区间 a, b （标准化后）
        ax = (x_range[0] - cx) / std_dev
        bx = (x_range[1] - cx) / std_dev
        ay = (y_range[0] - cy) / std_dev
        by = (y_range[1] - cy) / std_dev

        x = truncnorm.rvs(ax, bx, loc=cx, scale=std_dev)
        y = truncnorm.rvs(ay, by, loc=cy, scale=std_dev)
        points.append((x, y))

    return points


# === 使用示例 ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 500  # 点总数
    k = 4  # 簇数
    pts = generate_clustered_points_trunc(n, k,
                                          x_range=(5, 15),
                                          y_range=(0, 7.5),
                                          std_dev=0.8)
    xs, ys = zip(*pts)

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=15, alpha=0.6)
    plt.xlim(5, 15)
    plt.ylim(0, 7.5)
    plt.title(f"{k} clusters (truncated), n={n}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()
