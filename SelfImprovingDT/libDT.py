import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def plot_delaunay(points, tri):
    """绘制Delaunay三角剖分"""
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Delaunay Triangulation")
    plt.show()


if __name__ == "__main__":
    import time


    def timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print("{} 函数运行时间：{:.6f}秒".format(func.__name__, end_time - start_time))
            return result

        return wrapper


    @timer
    def main():
        # 生成随机点
        np.random.seed(42)
        points = np.random.rand(100, 2)

        # 计算Delaunay三角剖分
        tri = Delaunay(points)
        print(tri)
        # 绘制结果
        # plot_delaunay(points, tri)

    main()
