import math
import numpy as np

from RandomPointsInTrian import random_points_in_triangle
from DCEL.vertex import Vertex
from base.Trainglation.TrainClassic import TrainClassic
from DCEL.dcel import DCEL
from GlobalTestDelunay import GlobalTestDelunay

def main():
    # 创建 DCEL 实例
    dcel = TrainClassic()
    # dcel = DCEL()
    Scale = 1

    num = 9
    v1 = dcel.add_vertex(0.0 * Scale, 0.0 * Scale)
    v2 = dcel.add_vertex(1.0 * Scale, 0.0 * Scale)
    v3 = dcel.add_vertex(0.5 * Scale, math.sqrt(3)/2 * Scale)

    vertexs = [Vertex(0.5, 0.3),
               Vertex(0.3, 0.4),
               Vertex(0.4, 0.1),
               Vertex(0.6, 0.4),
               Vertex(0.3,0.2),
               Vertex(0.5,0.45),
               Vertex(0.6,0.2),
               Vertex(0.7,0.35),
               Vertex(0.7,0.1),]
    A = np.array([v1.x + Scale * 0.1, v1.y + Scale * 0.1])
    B = np.array([v2.x - Scale * 0.1, v2.y + Scale * 0.1])
    C = np.array([v3.x, v3.y - Scale * 0.1])

    vertexs_np = random_points_in_triangle(num,A,B,C)

    # 添加三个顶点（逆时针顺序）构造初始三角形
    initial_face = dcel.create_initial_triangle(v1, v2, v3)
    print(dcel)
    dcel.draw()

    for i in range(num):
        # point = Vertex(vertexs_np[i][0],vertexs_np[i][1])
        # print("Inserting:",point)
        # dcel.insert_point_with_certificate(point)
        dcel.insert_point_with_certificate(vertexs[i])
        # dcel.draw()
        # print(dcel)

    # dcel.insert_point_with_certificate(vertexs[0])
    # dcel.draw()
    # print(dcel)
    # dcel.insert_point_with_certificate(vertexs[1])
    # dcel.draw()
    # print(dcel)
    # dcel.insert_point_with_certificate(vertexs[2])
    # dcel.draw_edge_index()
    # print(dcel)

    dcel.draw_science()
    dcel.draw_science(True,False,False,True,True)

    print(GlobalTestDelunay(dcel))

if __name__ == "__main__":
    # VertexTest()
    main()
