from TrainClassic import TrainClassic
from base.DCEL.face import Face
from base.DCEL.vertex import Vertex
from base.HistoricalDAG.HistoryDAG import HistoryDAG

class TraingleBowyer(TrainClassic):
    """

    """
    def __init__(self):
        super().__init__()
        self.history = HistoryDAG()

    def locate_face(self, x: float, y: float) -> Face:
        # 使用历史DAG来定位面
        pass

    def insert_point_in_triangle(self, face: Face, x: float, y: float):
        pass

    def add_edge(self, point1: Vertex, point2: Vertex, face = None):
        pass

    def remove_edge(self, point1: Vertex, point2: Vertex):
        pass

    def insert_point_with_certificate(self, point: Vertex):
        pass






