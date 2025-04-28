import numpy as np

def random_points_in_triangle(n, A, B, C):
    """
    在 △ABC 内均匀生成 n 个随机点
    """
    np.random.seed(42)
    u = np.random.rand(n)
    v = np.random.rand(n)
    # 反射法把 (u,v) 保持在 u+v<=1 的区域
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    # 仿射组合
    return A + u[:, None] * (B - A) + v[:, None] * (C - A)
