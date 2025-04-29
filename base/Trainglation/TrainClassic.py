from base.DCEL.dcel import DCEL
from base.DCEL.face import Face
from base.DCEL.vertex import Vertex
from base.DCEL.halfedge import HalfEdge


class TrainClassic(DCEL):
    # def locate_face(self, x: float, y: float) -> Face:
    #     pass

    # def add_edge(self):
    #     pass

    # def remove_edge(self, edge:HalfEdge):
    #
    def connected_point(self,face: Face, point: Vertex):
        start = face.outer_component
        he = start
        while True:

            if he == start:
                break
        pass
    def insert_point_with_certificate(self, point: Vertex):

        face = self.locate_face(point.x, point.y)
        half_edges_stacks = self.enumerate_half_edges(face)

        self.insert_point_in_triangle(face,point.x,point.y)
        while True:
            if len(half_edges_stacks) == 0:
                break
            for he in half_edges_stacks:
                he.certificate = self.isDelunayTrain(he,point)
                if he.certificate:
                    # hePointA = he.origin
                    # hePointB = he.next.origin
                    # oppositePoint = he.twin.prev.origin
                    half_edges_stacks.append(he.twin.prev)
                    half_edges_stacks.append(he.twin.next)
                    half_edges_stacks.remove(he)
                    self.filp_edge(he,point)
                    # self.remove_edge(hePointA,hePointB)
                else:
                    half_edges_stacks.remove(he)

        return