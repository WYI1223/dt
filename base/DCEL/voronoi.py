from base.DCEL.vertex import Vertex
from base.DCEL.halfedge import HalfEdge
from base.DCEL.face import Face
from base.DCEL.geometry import orientation, in_circle_test, midpoint
from base.DCEL.dcel import DCEL

import numpy as np

class Voronoi:

    def __init__(self):
        self.vertices = []    # 所有顶点
        self.half_edges = []  # 所有半边
        self.faces = []       # 所有面（有限面）
        self.outer_face = None  # 无限面

    import numpy as np

    def circumcenters_batch(pts):
        """
        批量计算三点外接圆圆心。

        参数
        ----
        pts : ndarray, shape (N, 3, 2)
            N 组三角形顶点坐标，每组三个二维点 (x, y)。

        返回
        ----
        centers : ndarray, shape (N, 2)
            每组三点对应的圆心坐标 (xc, yc)。
            若三点（近似）共线，则返回 NaN。
        """
        pts = np.asarray(pts, dtype=float)
        if pts.shape[-2:] != (3, 2):
            raise ValueError("输入必须形如 (N, 3, 2)")

        # 拆分坐标
        x = pts[..., 0]  # (N, 3)
        y = pts[..., 1]  # (N, 3)

        # 三点平方和
        x2_y2 = x ** 2 + y ** 2  # (N, 3)

        # 计算两倍有向面积 d = 2*Δ  （行列式式子，沿最后一轴取索引）
        d = 2 * (
                x[:, 0] * (y[:, 1] - y[:, 2]) +
                x[:, 1] * (y[:, 2] - y[:, 0]) +
                x[:, 2] * (y[:, 0] - y[:, 1])
        )  # (N,)

        # 防止除零：|d| 很小视为共线
        mask = np.abs(d) < 1e-12
        d[mask] = np.nan  # 先置为 NaN，后面除法安全

        # 圆心公式（向量化）
        xc = (
                     x2_y2[:, 0] * (y[:, 1] - y[:, 2]) +
                     x2_y2[:, 1] * (y[:, 2] - y[:, 0]) +
                     x2_y2[:, 2] * (y[:, 0] - y[:, 1])
             ) / d

        yc = (
                     x2_y2[:, 0] * (x[:, 2] - x[:, 1]) +
                     x2_y2[:, 1] * (x[:, 0] - x[:, 2]) +
                     x2_y2[:, 2] * (x[:, 1] - x[:, 0])
             ) / d

        centers = np.stack([xc, yc], axis=-1)  # (N, 2)
        return centers


    def delaunay2voronoi(self, triangulation:DCEL):


        triangles = triangulation.faces
        he_Q = triangulation.half_edges

        pts_raw = []

        for triangle in triangles:
            pts_raw.append(triangulation.enumerate_vertices(triangle))

        pts_np = np.array(
            [[(v.x, v.y) for v in tri] for tri in pts_raw],
            dtype=float,  # 强制转成浮点
        )

        self.vertices = self.circumcenters_batch(pts_np)

        # 这里应该将triangle写成一个dequeue队列，一个一个取出并将he转为voronoi的he
        # 使用完一个之后dequeue再继续取下一个
        for triangle in triangles:
            dt_hes = triangulation.enumerate_half_edges(triangle)




        return


