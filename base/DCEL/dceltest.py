import math
import random

from vertex import Vertex
from dcel import DCEL


def main():
    # 创建 DCEL 实例
    dcel = DCEL()

    # 添加三个顶点（逆时针顺序）构造初始三角形
    v1 = dcel.add_vertex(0.0, 0.0)
    v2 = dcel.add_vertex(1.0, 0.0)
    v3 = dcel.add_vertex(0.5, math.sqrt(3)/2)
    initial_face = dcel.create_initial_triangle(v1, v2, v3)
    print("初始 DCEL 结构：")
    print(dcel)

    print("绘制初始三角形...")
    dcel.draw()

    # 在初始三角形中插入新点
    new_x, new_y = 0.5, 0.3
    new_faces = dcel.insert_point_in_triangle(initial_face, new_x, new_y)
    print("插入新点后生成的面：")
    for face in new_faces:
        print("面：", face)
        print("顶点：", dcel.enumerate_vertices(face))

    print("绘制更新后 DCEL 结构...")
    dcel.draw()

    # 可继续插入点并展示 locate_face 函数效果
    dcel.insert_point_in_triangle(dcel.faces[2], 0.4, 0.4)
    dcel.draw()

    print(dcel.faces.index(dcel.locate_face(0.4, 0.1)))
    dcel.insert_point_in_triangle(dcel.locate_face(0.4, 0.1), 0.4, 0.1)
    dcel.draw()

    print(dcel.faces.index(dcel.locate_face(0.6, 0.4)))
    dcel.insert_point_in_triangle(dcel.locate_face(0.6, 0.4), 0.6, 0.4)
    dcel.draw()

    print(dcel.find_halfedges(Vertex(0.5, 0.3),dcel.vertices[4]))
    dcel.remove_edge(dcel.vertices[3],dcel.vertices[4])
    dcel.draw()
    print(dcel)
    # dcel.remove_edge(dcel.vertices[3],dcel.vertices[2])
    # dcel.draw()
    # print(dcel)
    # dcel.remove_edge(dcel.vertices[3], dcel.vertices[6])
    # dcel.draw()
    # print(dcel)
    # dcel.add_edge(dcel.vertices[4], dcel.vertices[3])
    # dcel.draw()
    # print(dcel)
    # dcel.add_edge(dcel.vertices[3], dcel.vertices[6])
    # dcel.draw()
    # print(dcel)
    # dcel.add_edge(dcel.vertices[4], dcel.vertices[6])
    # dcel.draw()
    # print(dcel)
    dcel.remove_edge(dcel.vertices[3], dcel.vertices[2])
    dcel.draw()
    print(dcel)
    dcel.add_edge(dcel.vertices[0], dcel.vertices[6],dcel.faces[1])
    dcel.draw()
    print(dcel)

def add_test():
    # 创建 DCEL 实例
    dcel = DCEL()

    # 添加三个顶点（逆时针顺序）构造初始三角形
    v1 = dcel.add_vertex(0.0, 0.0)
    v2 = dcel.add_vertex(1.0, 0.0)
    v3 = dcel.add_vertex(0.5, math.sqrt(3)/2)
    initial_face = dcel.create_initial_triangle(v1, v2, v3)
    print("初始 DCEL 结构：")
    print(dcel)

    print("绘制初始三角形...")
    dcel.draw()

    v4 = dcel.add_vertex(0.5, 0.3)
    dcel.add_edge(v1,v4)
    dcel.draw()

def VertexTest():
    vertex = Vertex(0.5, 0.3)
    point = Vertex(0.5, 0.3)
    print(vertex == point)
if __name__ == "__main__":
    # VertexTest()
    # main()
    add_test()
