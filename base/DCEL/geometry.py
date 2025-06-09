import math
import numpy as np
from base.DCEL.vertex import Vertex
def orientation(p, q, r):
    """
    计算向量 (q - p) 与 (r - p) 的叉积；
    若结果 > 0，则 r 在有向线段 p->q 的左侧，
    若结果 < 0，则 r 在其右侧；
    若结果 = 0，则三点共线。
    计算三点 p, q, r 的方向关系，利用叉积公式：
          ToLeft(p, q, r) = | p.x p.y 1 |
                            | q.x q.y 1 | = (q.x - p.x)*(r.y - p.y) - (q.y - p.y)*(r.x - p.x)
                            | r.x r.y 1 |
        返回值 > 0 表示 r 在向量 p->q 的左侧，
               = 0 表示共线，
               < 0 表示在右侧。
    """
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

def in_circle_test(p, q, r, d):
    """
    计算 4x4 行列式（InCircle 测试）
    对于逆时针顺序的 p, q, r 来说：
      如果结果 > 0，则 d 落在由 p, q, r 构成的圆内。
    """
    M = np.array([
        [p.x, p.y, p.x**2 + p.y**2, 1],
        [q.x, q.y, q.x**2 + q.y**2, 1],
        [r.x, r.y, r.x**2 + r.y**2, 1],
        [d.x, d.y, d.x**2 + d.y**2, 1]
    ])
    return np.linalg.det(M)

def midpoint(p):
    """
    input a array of Vertex, then calculate the midpoint of all vertexs
    :param p:
    :return:
    """
    sum_X = 0
    sum_Y = 0
    num = len(p)
    for i in p:
        sum_X += i.x
        sum_Y += i.y
    mid = Vertex(sum_X/num, sum_Y/num)
    return mid


def circumcenter(a,b,c):
    """
    计算三角形 ABC 的外接圆圆心。

    参数
    ----
    a, b, c : (x, y)
        三角形三个顶点的笛卡尔坐标。

    返回
    ----
    (cx, cy) : (float, float)
        外接圆圆心坐标。

    抛出
    ----
    ValueError
        当三点共线（无法唯一确定外接圆）时抛出。
    """
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y

    # 行列式分母：2·(ax(by−cy)+bx(cy−ay)+cx(ay−by))
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:  # 共线或几乎共线
        raise ValueError("Points are colinear; circumcenter is undefined.")

    # 使用坐标法推导得到的分子
    ax2_ay2 = ax ** 2 + ay ** 2
    bx2_by2 = bx ** 2 + by ** 2
    cx2_cy2 = cx ** 2 + cy ** 2

    ux = (ax2_ay2 * (by - cy) +
          bx2_by2 * (cy - ay) +
          cx2_cy2 * (ay - by)) / d

    uy = (ax2_ay2 * (cx - bx) +
          bx2_by2 * (ax - cx) +
          cx2_cy2 * (bx - ax)) / d

    return ux, uy