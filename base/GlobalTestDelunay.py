from base.DCEL.dcel import DCEL
from base.DCEL.geometry import in_circle_test

def GlobalTestDelunay(dcel:DCEL):

    Triangles = dcel.faces
    Points = dcel.vertices

    for trinagle in Triangles:
        vertexs = dcel.enumerate_vertices(trinagle)
        for point in Points:
            A = vertexs[0]
            B = vertexs[1]
            C = vertexs[2]
            if point in vertexs:
                continue
            if in_circle_test(A,B,C,point) > 0:
                print("Triangle:", dcel.vertices.index(A), "",
                      dcel.vertices.index(B), "",
                      dcel.vertices.index(C), "are INCLUDE a Point:",
                      dcel.vertices.index(point),in_circle_test(A,B,C,point)
                      )
                return False
    return True
def main():

    return

if __name__ == '__main__':
    main()