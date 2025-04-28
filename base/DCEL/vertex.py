class Vertex:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.incident_edge = None  # 任一与该顶点关联的半边

    def __repr__(self):
        return f"Vertex({self.x:.2f}, {self.y:.2f})"

    def __eq__(self, other):
        # if not isinstance(other, Vertex):
        #     return NotImplemented
        # 对于浮点数的比较，可以考虑使用 math.isclose 来处理精度问题
        import math
        return math.isclose(self.x, other.x, rel_tol=1e-9) and math.isclose(self.y, other.y, rel_tol=1e-9)

