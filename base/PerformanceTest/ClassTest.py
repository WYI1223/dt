# import time
# import math
# import numpy as np
# from base.DCEL.dcel import DCEL
# from base.DCEL.vertex import Vertex
# from base.RandomPointsInTrian import random_points_in_triangle

# def main():
#
#     dcel = DCEL()
#     Scale = 10
#     v1 = dcel.add_vertex(0.0 * Scale, 0.0 * Scale)
#     v2 = dcel.add_vertex(1.0 * Scale, 0.0 * Scale)
#     v3 = dcel.add_vertex(0.5 * Scale, math.sqrt(3)/2 * Scale)
#
#     vertexs = [Vertex(0.5, 0.3),
#                Vertex(0.3, 0.4),
#                Vertex(0.4, 0.1),
#                Vertex(0.6, 0.4),
#                Vertex(0.3,0.2)]
#     A = np.array([v1.x + Scale * 0.1, v1.y + Scale * 0.1])
#     B = np.array([v2.x - Scale * 0.1, v2.y + Scale * 0.1])
#     C = np.array([v3.x, v3.y - Scale * 0.1])
#
#     # vertexs_np = poisson_disk_triangle(A,B,C,0.1,10)
#
#     vertexs_np = random_points_in_triangle(20,A,B,C)
#
#     # 添加三个顶点（逆时针顺序）构造初始三角形
#     initial_face = dcel.create_initial_triangle(v1, v2, v3)
#
#     start_time = time.perf_counter()
#
#     for e in vertexs_np:
#         point = Vertex(e[0],e[1])
#         dcel.insert_point_with_certificate(point)
#
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time
#
#     dcel.draw()
#
#     print(elapsed_time)
#
#     return
#

import time
import math
import numpy as np
import pandas as pd
from base.Trainglation.TrainClassic import TrainClassic
from base.RandomPointsInTrian import random_points_in_triangle
from base.DCEL.vertex import Vertex

def benchmark_dcel(ns, scale=1.0, csv_filename="benchmark_results.csv"):
    results = []

    for n in ns:
        # Prepare DCEL and initial triangle
        dcel = TrainClassic()
        v1 = dcel.add_vertex(0.0 * scale, 0.0 * scale)
        v2 = dcel.add_vertex(1.0 * scale, 0.0 * scale)
        v3 = dcel.add_vertex(0.5 * scale, math.sqrt(3) / 2 * scale)
        A = np.array([v1.x + scale * 0.1, v1.y + scale * 0.1])
        B = np.array([v2.x - scale * 0.1, v2.y + scale * 0.1])
        C = np.array([v3.x, v3.y - scale * 0.1])
        points = random_points_in_triangle(n, A, B, C)
        dcel.create_initial_triangle(v1, v2, v3)

        # Benchmark insertion
        start = time.perf_counter()
        for pt in points:
            vertex = Vertex(pt[0], pt[1])
            dcel.insert_point_with_certificate(vertex)
        end = time.perf_counter()

        elapsed = end - start
        print(f"n={n}: {elapsed:.6f} seconds")
        results.append({"n": n, "time_s": elapsed})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"Benchmark results saved to {csv_filename}")


if __name__ == "__main__":
    ns = [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    benchmark_dcel(ns)
#
# if __name__ == '__main__':
#     main()