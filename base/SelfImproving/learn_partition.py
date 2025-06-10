# learn_partition.py

import random
import math


# def learn_approx_partition(train_samples):
#     """
#     用 k-means 对每个下标 i 构造特征，再做近似分区。
#
#     输入：
#       train_samples: 训练样本列表，每个样本是一个点集 I=[(x₀,y₀),(x₁,y₁),...,(x_{n−1},y_{n−1})]
#                      假定所有样本大小都相同 = n。
#     输出：
#       Gp: 一个列表 [G'_0, G'_1, ..., G'_K]：
#         - G'_0 = []（留空常数组）
#         - 对 j=1..K, G'_j 是那些在 k-means 聚到第 j−1 类的下标 i 列表。
#     方法：
#       1) 先计算每个 “下标 i” 在 m 个样本中的平均坐标 (avg_x, avg_y)；
#       2) 令 K = max(1, ⌊√n⌋)；
#       3) 用随机选的 K 个中心初始化 k-means；
#       4) 迭代：每个 i 根据最近中心归入某个簇；然后重算 K 个中心；
#          最多 100 轮或直到簇不再变化。
#       5) 返回 Gp：Gp[0]=[], Gp[j] = { all i | clusters[i]==j−1 }。
#     """
#
#     if not train_samples:
#         return []
#     n = len(train_samples[0])  # 样本大小 = n
#     m = len(train_samples)  # 训练样本个数
#
#     # 1) 计算 avg_coords[i] = (平均 x, 平均 y) over all m 个样本
#     avg_coords = []
#     for i in range(n):
#         sum_x = sum(train_samples[s][i][0] for s in range(m))
#         sum_y = sum(train_samples[s][i][1] for s in range(m))
#         avg_coords.append((sum_x / m, sum_y / m))
#
#     # 2) 确定簇数 K ≈ √n
#     K = max(1, int(math.sqrt(n)))
#
#     # 3) 初始化：随机在 n 个 avg_coords 里抽 K 个做初始质心
#     indices = list(range(n))
#     random.shuffle(indices)
#     centroids = [avg_coords[indices[i]] for i in range(K)]
#
#     # clusters[i] 保存 avg_coords[i] 属于哪一簇 (0..K−1)
#     clusters = [0] * n
#
#     # 4) 迭代 k-means，最多 100 轮
#     for _ in range(100):
#         changed = False
#
#         # 4a) Assignment：每个 i 找最近的 center
#         new_clusters = []
#         for i in range(n):
#             x_i, y_i = avg_coords[i]
#             best_j = 0
#             dx0 = x_i - centroids[0][0]
#             dy0 = y_i - centroids[0][1]
#             best_dist = dx0 * dx0 + dy0 * dy0
#             for j in range(1, K):
#                 dx = x_i - centroids[j][0]
#                 dy = y_i - centroids[j][1]
#                 d2 = dx * dx + dy * dy
#                 if d2 < best_dist:
#                     best_dist = d2
#                     best_j = j
#             new_clusters.append(best_j)
#             if new_clusters[-1] != clusters[i]:
#                 changed = True
#
#         clusters = new_clusters
#         if not changed:
#             break
#
#         # 4b) Update centroids：重算每一簇的平均坐标
#         sums = [(0.0, 0.0, 0) for _ in range(K)]
#         for i in range(n):
#             j = clusters[i]
#             sx, sy, cnt = sums[j]
#             x_i, y_i = avg_coords[i]
#             sums[j] = (sx + x_i, sy + y_i, cnt + 1)
#         for j in range(K):
#             sx, sy, cnt = sums[j]
#             if cnt > 0:
#                 centroids[j] = (sx / cnt, sy / cnt)
#             else:
#                 # 若某簇空了，就随机重置一个质心
#                 centroids[j] = avg_coords[random.randrange(n)]
#
#     # 5) 构造输出 Gp
#     Gp = [[] for _ in range(K + 1)]
#     Gp[0] = []  # 常数组留空
#     for i in range(n):
#         # clusters[i] ∈ [0..K−1] → 放到 Gp[clusters[i]+1]
#         Gp[clusters[i] + 1].append(i)
#
#     return Gp

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def learn_approx_partition(train_samples, K):
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

if __name__ == '__main__':
    train_samples = [
        [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    ]
    Gp = learn_approx_partition(train_samples,3)
    print(Gp)
    # 例如输出：[[], [0, 1], [2,3,4]] （具体依随机初始而定）
