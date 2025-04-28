from base.DCEL.vertex import Vertex

class HalfEdge:
    def __init__(self, origin: Vertex):
        self.origin = origin          # 起始顶点
        self.twin = None              # 对边
        self.next = None              # 同一面中下一半边
        self.prev = None              # 同一面中上一半边
        self.incident_face = None     # 所属面

        # 额外字段：例如 certificate 或其他辅助信息可在此扩展
        self.certificate = False

    def __repr__(self):
        return f"HalfEdge({self.origin})"
