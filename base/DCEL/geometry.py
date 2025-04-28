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