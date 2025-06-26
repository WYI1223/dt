import numpy as np
import matplotlib.pyplot as plt

from base.SelfImproving.PointDistribution2 import generate_clustered_points_superfast

# 1) 生成样本
pts = np.array(generate_clustered_points_superfast(
    n=20000, k=3,
    x_range=(5,15), y_range=(0,7.5),
    std_dev=0.8, layout="grid"
))

# 2) 计算 2D 直方图，开启 density=True 以归一化为“概率密度”
x_edges = np.linspace(5, 15, 100)
y_edges = np.linspace(0, 7.5,  100)
H, x_edges, y_edges = np.histogram2dw(
    pts[:,0], pts[:,1],
    bins=[x_edges, y_edges],
    density=True
)

# 3) 中心点网格
X, Y = np.meshgrid(
    0.5*(x_edges[:-1] + x_edges[1:]),
    0.5*(y_edges[:-1] + y_edges[1:])
)

# 4) 绘图
plt.figure(figsize=(6,4))
plt.pcolormesh(X, Y, H.T)    # 或者 plt.imshow(H.T, extent=[5,15,0,7.5], origin='lower')
plt.colorbar(label='Estimated density')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Heatmap of Generated Distribution')
plt.tight_layout()
plt.show()
