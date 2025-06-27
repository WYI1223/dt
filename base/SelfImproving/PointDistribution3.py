import numpy as np
from itertools import zip_longest

def generate_clustered_points_superfast_order(
    n, k,
    x_range=(5, 15),
    y_range=(0, 7.5),
    std_dev=0.4,
    weights=None,
    layout="x_split",   # 可选 "x_split" 或 "grid"
):
    """
    超快的“有规律”聚类点生成器 —— 全矢量化 + 边界 clip + 簇间交替输出。

    参数
    ----
    n : int
        需要生成的点总数
    k : int
        簇数
    x_range, y_range : tuple(float, float)
        生成区域 [xmin, xmax] × [ymin, ymax]
    std_dev : float
        每个簇的标准差（控制簇大小）
    weights : None 或长度 k 的数组
        各簇被选中概率；None 则均匀
    layout : str
        簇中心布局方式： "x_split" 或 "grid"
    返回
    ----
    points : list[(float, float)]
        以 cluster0, cluster1, …, cluster(k-1), cluster0, cluster1, … 轮询顺序输出的 n 个聚类点
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    # 1) 生成有规律的中心数组 centers.shape == (k,2)
    if layout == "x_split":
        xs = np.linspace(xmin, xmax, k+2)[1:-1]
        ys = np.full(k, 0.5*(ymin + ymax))
        centers = np.stack([xs, ys], axis=1)
    elif layout == "grid":
        kx = int(np.ceil(np.sqrt(k)))
        ky = int(np.ceil(k / kx))
        xs = np.linspace(xmin, xmax, kx+1)[:-1] + (xmax-xmin)/(2*kx)
        ys = np.linspace(ymin, ymax, ky+1)[:-1] + (ymax-ymin)/(2*ky)
        grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
        centers = grid[:k]
    else:
        raise ValueError("layout 必须是 'x_split' 或 'grid'")

    # 2) 按 weights 生成标签
    if weights is None:
        weights = np.ones(k) / k
    labels = np.random.choice(k, size=n, p=weights)

    # 3) 矢量化采样 + 偏移 + clip
    noise = np.random.normal(size=(n, 2)) * std_dev
    centers_per_point = centers[labels]
    pts = centers_per_point + noise
    pts[:, 0] = np.clip(pts[:, 0], xmin, xmax)
    pts[:, 1] = np.clip(pts[:, 1], ymin, ymax)

    # 4) 按簇分组
    clustered = [pts[labels == i] for i in range(k)]

    # 5) 轮询（round-robin）依次取每个簇的点
    interleaved = []
    # zip_longest 会用 None 填充短列，我们过滤掉 None
    for group in zip_longest(*clustered, fillvalue=None):
        for pt in group:
            if pt is not None:
                interleaved.append((float(pt[0]), float(pt[1])))

    return interleaved

# === 使用示例 ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n, k = 300, 3
    pts = generate_clustered_points_superfast_order(
        n, k,
        x_range=(5, 15),
        y_range=(0, 7.5),
        std_dev=0.4,
        layout="x_split"
    )
    xs, ys = zip(*pts)
    plt.figure(figsize=(6,4))
    plt.scatter(xs, ys, c=[i % k for i in range(len(xs))], s=10, alpha=0.6)
    plt.xlim(5,15); plt.ylim(0,7.5)
    plt.title("Round‐robin clustered points (3 clusters)")
    plt.show()
