import math
import numpy as np

from base.RandomPointsInTrian import random_points_in_triangle
from base.DCEL.vertex import Vertex
from base.Trainglation.TrainClassic import TrainClassic
from base.GlobalTestDelunay import GlobalTestDelunay

def compute_delaunay(vertexs):
    # 创建 DCEL 实例
    dcel = TrainClassic()
    # dcel = DCEL()
    Scale = 20

    num = len(vertexs)
    # num = 9
    v1 = dcel.add_vertex(0.0 * Scale, 0.0 * Scale)
    v2 = dcel.add_vertex(1.0 * Scale, 0.0 * Scale)
    v3 = dcel.add_vertex(0.5 * Scale, math.sqrt(3)/2 * Scale)
    A = np.array([v1.x + Scale * 0.1, v1.y + Scale * 0.1])
    B = np.array([v2.x - Scale * 0.1, v2.y + Scale * 0.1])
    C = np.array([v3.x, v3.y - Scale * 0.1])

    vertexs_np = random_points_in_triangle(num,A,B,C)

    # 添加三个顶点（逆时针顺序）构造初始三角形
    initial_face = dcel.create_initial_triangle(v1, v2, v3)
    # print(dcel)
    # dcel.draw()
    #
    for i in range(num):
        # point = Vertex(vertexs_np[i][0],vertexs_np[i][1])
        # print("Inserting:",point)
        # dcel.insert_point_with_certificate(point)
        dcel.insert_point_with_certificate(vertexs[i])
        # dcel.draw()
        # print(dcel)


    print(GlobalTestDelunay(dcel))
    return dcel

if __name__ == "__main__":
    # VertexTest()
    vertexs = [Vertex(0.5, 0.3),
               Vertex(0.3, 0.4),
               Vertex(0.4, 0.1),
               Vertex(0.6, 0.4),
               Vertex(0.3,0.2),
               Vertex(0.5,0.45),
               Vertex(0.6,0.2),
               Vertex(0.7,0.35),
               Vertex(0.7,0.1),]

    dcel = compute_delaunay(vertexs)
    # dcel.draw()
    dcel.draw_science()
    # dcel.draw_science(True,False,False,True,True)
