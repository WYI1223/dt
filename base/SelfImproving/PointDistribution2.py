import numpy as np

def generate_clustered_points_superfast(
    n, k,
    x_range=(5, 15),
    y_range=(0, 7.5),
    std_dev=0.5,
    weights=None,
    layout="x_split",   # 可选 "x_split" 或 "grid"
):
    """
    超快的“有规律”聚类点生成器 —— 全矢量化 + 边界 clip。

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
        簇中心布局方式：
          - "x_split"：x 轴等距切分，y 取中点
          - "grid"  ：在区域内做近似 √k×√k 网格，依次取前 k 个格子中心
    返回
    ----
    points : list[(float, float)]
        生成的 n 个聚类点
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    # 1) 生成有规律的中心数组 centers.shape == (k,2)
    if layout == "x_split":
        xs = np.linspace(xmin, xmax, k+2)[1:-1]  # 去掉两端，得到 k 个中点
        ys = np.full(k, 0.5*(ymin + ymax))
        centers = np.stack([xs, ys], axis=1)
    elif layout == "grid":
        # 找到接近的网格大小
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

    # 3) 矢量化：对所有点一次性採样 + 偏移 + clip
    #   (n,2) 的标准正态
    noise = np.random.normal(size=(n, 2)) * std_dev
    # 每个点对应的中心
    centers_per_point = centers[labels]
    pts = centers_per_point + noise
    pts[:, 0] = np.clip(pts[:, 0], xmin, xmax)
    pts[:, 1] = np.clip(pts[:, 1], ymin, ymax)

    return pts.tolist()

# === 使用示例 ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n, k = 10000, 3
    pts = generate_clustered_points_superfast(
        n, k,
        x_range=(5, 15),
        y_range=(0, 7.5),
        std_dev=0.6,
        layout="grid"
    )
    xs, ys = zip(*pts)
    plt.figure(figsize=(6,4))
    plt.scatter(xs, ys, s=5, alpha=0.5)
    # 画出各簇中心
    # (仅作示意，不必在最终代码里)
    # centers = np.array(generate_clustered_points_superfast.__defaults__[0])
    plt.xlim(5,15); plt.ylim(0,7.5)
    plt.title("Superfast clustered points")
    plt.show()
