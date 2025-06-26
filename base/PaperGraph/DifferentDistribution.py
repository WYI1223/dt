import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection

def plot_delaunay(ax, points, xlim=(0,1), ylim=(0,1), **kwargs):
    # Delaunay 三角剖分
    tri = Delaunay(points)
    lines = []
    for simplex in tri.simplices:
        a, b, c = simplex
        lines.append(points[[a, b]])
        lines.append(points[[b, c]])
        lines.append(points[[c, a]])
    # 绘制
    lc = LineCollection(lines, **kwargs)
    ax.add_collection(lc)
    ax.scatter(points[:, 0], points[:, 1], s=12, color='blue', zorder=2)
    # 固定坐标轴范围
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')

def generate_uniform(n=200):
    return np.random.rand(n, 2)

def generate_gaussian(n=200, mean=(0.5, 0.5), std=0.1):
    return np.random.randn(n, 2) * std + mean

def generate_gaussian_mixture(n_per_cluster=70):
    centers = np.array([
        [0.5, 0.5 + 0.3/np.sqrt(3)],
        [0.5 - 0.15, 0.5 - 0.3/(2*np.sqrt(3))],
        [0.5 + 0.15, 0.5 - 0.3/(2*np.sqrt(3))]
    ])
    std = 0.05
    pts = [np.random.randn(n_per_cluster, 2) * std + c for c in centers]
    return np.vstack(pts)

def main():
    np.random.seed(42)
    pts_uni   = generate_uniform()
    pts_gauss = generate_gaussian()
    pts_mix   = generate_gaussian_mixture()

    # 把高度调大到 7 英寸，让标题有足够空间
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    titles = ['Uniform Random Points',
              'Single Gaussian Cluster',
              'Gaussian Mixture (3 clusters)']
    datasets = [pts_uni, pts_gauss, pts_mix]

    for ax, pts, title in zip(axes, datasets, titles):
        plot_delaunay(ax, pts,
                      xlim=(0, 1), ylim=(0, 1),
                      colors='gray', linewidths=0.6)
        # 统一标题与子图的距离
        ax.set_title(title, fontsize=16, pad=20)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
