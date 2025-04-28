import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class Triangulation:
    def __init__(self):
        # Infinite bounds for the super triangle
        self.triangles = [(0, 1, 2)]
        self.vertices = [
            np.array([-500, -500]),
            np.array([500, -500]),
            np.array([0, 500])
        ]

    def add_point(self, p):
        p = np.array(p)
        self.vertices.append(p)
        pi = len(self.vertices) - 1

        bad_triangles = []
        for i, (a, b, c) in enumerate(self.triangles):
            if self.in_circle(self.vertices[a], self.vertices[b], self.vertices[c], p):
                bad_triangles.append(i)

        # Remove bad triangles
        edges = []
        for i in reversed(bad_triangles):
            a, b, c = self.triangles.pop(i)
            edges += [(a, b), (b, c), (c, a)]

        # Deduplicate edges
        unique_edges = []
        for edge in edges:
            if edges.count(edge) == 1 and edges.count(edge[::-1]) == 0:
                unique_edges.append(edge)

        # Re-triangulate the polygonal hole
        for a, b in unique_edges:
            self.triangles.append((a, b, pi))

    def in_circle(self, a, b, c, p):
        m = np.array([
            [a[0], a[1], a[0]**2 + a[1]**2, 1],
            [b[0], b[1], b[0]**2 + b[1]**2, 1],
            [c[0], c[1], c[0]**2 + c[1]**2, 1],
            [p[0], p[1], p[0]**2 + p[1]**2, 1]
        ])
        return np.linalg.det(m) > 0

    def get_triangles(self):
        real_triangles = []
        for a, b, c in self.triangles:
            if a > 2 and b > 2 and c > 2:
                real_triangles.append((a, b, c))
        return real_triangles

    def plot(self):
        plt.figure(figsize=(8, 8))
        for a, b, c in self.triangles:
            if a > 2 and b > 2 and c > 2:
                triangle = np.array([self.vertices[a], self.vertices[b], self.vertices[c]])
                polygon = Polygon(triangle, edgecolor='black', fill=None, linewidth=1)
                plt.gca().add_patch(polygon)
        # Plot points on top
        for p in self.vertices[3:]:  # skip super triangle vertices
            plt.scatter(p[0], p[1], color='red')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()

def main():
    np.random.seed(0)
    points = np.random.rand(100, 2) * 100  # generate 100 random points
    triangulation = Triangulation()
    for point in points:
        triangulation.add_point(point)
    triangulation.plot()

if __name__ == "__main__":
    main()
