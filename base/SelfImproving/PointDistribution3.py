import numpy as np
from itertools import zip_longest

def generate_clustered_points_no_overlap(
    n, k,
    x_range=(5, 15),
    y_range=(0, 7.5),
    std_dev=0.4,
    weights=None,
    layout="x_split"
):
    """
    生成明显分区、无重叠的聚类点，并返回对应簇标签：
      1) 区域切分成互不重叠子块
      2) 在各自子块内生成高斯分布点
      3) 对点进行边界 clip
      4) 按簇交替输出，并附带每点的簇标签

    参数
    ----
    n : int
        点总数
    k : int
        簇数
    x_range, y_range : tuple
        区域范围 [xmin,xmax] × [ymin,ymax]
    std_dev : float
        每簇高斯噪声标准差
    weights : None 或长度 k 数组
        各簇采样概率；None 则均匀
    layout : str
        子区域划分方式：
          - "x_split"：x 轴等宽切分
          - "grid"   ：近似 √k×√k 网格切分

    返回
    ----
    points : list of (float, float)
        按簇轮询顺序输出的聚类点列表
    labels : list of int
        对应每个点所属的簇索引
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    # 1) 切分子区域
    regions = []
    if layout == "x_split":
        width = (xmax - xmin) / k
        for i in range(k):
            x0 = xmin + i * width
            x1 = x0 + width
            regions.append((x0, x1, ymin, ymax))
    elif layout == "grid":
        kx = int(np.ceil(np.sqrt(k)))
        ky = int(np.ceil(k / kx))
        cell_w = (xmax - xmin) / kx
        cell_h = (ymax - ymin) / ky
        count = 0
        for iy in range(ky):
            for ix in range(kx):
                if count < k:
                    x0 = xmin + ix * cell_w
                    x1 = x0 + cell_w
                    y0 = ymin + iy * cell_h
                    y1 = y0 + cell_h
                    regions.append((x0, x1, y0, y1))
                    count += 1
    else:
        raise ValueError("layout must be 'x_split' or 'grid'")

    # 2) 分配簇标签
    if weights is None:
        weights = np.ones(k) / k
    labels_global = np.random.choice(k, size=n, p=weights)

    # 3) 在子区域内生成高斯点并 clip
    clustered = []
    for i in range(k):
        cnt = np.sum(labels_global == i)
        x0, x1, y0, y1 = regions[i]
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        noise = np.random.normal(size=(cnt, 2)) * std_dev
        pts = center + noise
        pts[:, 0] = np.clip(pts[:, 0], x0, x1)
        pts[:, 1] = np.clip(pts[:, 1], y0, y1)
        clustered.append(pts)

    # 4) 按簇轮询输出，并记录标签
    interleaved = []
    labels_interleaved = []
    for group in zip_longest(*clustered, fillvalue=None):
        for cluster_idx, pt in enumerate(group):
            if pt is not None:
                interleaved.append((float(pt[0]), float(pt[1])))
                labels_interleaved.append(cluster_idx)

    return interleaved, labels_interleaved

# === 使用示例 ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n, k = 3000, 3
    pts, cls = generate_clustered_points_no_overlap(
        n, k,
        x_range=(5, 15),
        y_range=(0, 7.5),
        std_dev=0.4,
        layout="x_split"
    )
    xs, ys = zip(*pts)
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, c=cls, s=10, alpha=0.6)
    plt.xlim(5, 15)
    plt.ylim(0, 7.5)
    plt.title("Gaussian clustered points, no overlap with correct coloring")
    plt.show()
