import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TKinter backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle


# Bowyer-Watson
class Triangulation:
    def __init__(self):
        # Infinite bounds for the super triangle
        self.triangles = [(0, 1, 2)]
        self.vertices = [
            np.array([-500, -500]),
            np.array([500, -500]),
            np.array([0, 500])
        ]
        # Track the current point and triangles involved in checks
        self.current_point_index = None
        self.current_bad_triangles = []

    def triangle_orientation(self, a, b, c):
        # Returns positive if counter-clockwise, negative if clockwise, zero if colinear
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def add_point(self, p):
        p = np.array(p)

        self.vertices.append(p)
        pi = len(self.vertices) - 1
        self.current_point_index = pi

        bad_triangles = []
        for i, (a, b, c) in enumerate(self.triangles):
            if self.in_circle(self.vertices[a], self.vertices[b], self.vertices[c], p):
                bad_triangles.append(i)
        self.current_bad_triangles = [self.triangles[i] for i in bad_triangles]

        edges = []
        for i in reversed(bad_triangles):
            a, b, c = self.triangles.pop(i)
            edges += [(a, b), (b, c), (c, a)]

        # Remove duplicate edges
        edge_count = {}
        for edge in edges:
            edge = tuple(sorted(edge))
            edge_count[edge] = edge_count.get(edge, 0) + 1
        unique_edges = [edge for edge, count in edge_count.items() if count == 1]

        for a, b in unique_edges:
            self.triangles.append((a, b, pi))

        self.plot(pause_time=0.5)  # Display updates after point is added

    def is_within_super_triangle(self, p):
        # Points of the super triangle
        A, B, C = self.vertices[0], self.vertices[1], self.vertices[2]

        # Barycentric coordinates to determine if the point is within the triangle
        detT = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        lambda1 = ((B[1] - C[1]) * (p[0] - C[0]) + (C[0] - B[0]) * (p[1] - C[1])) / detT
        lambda2 = ((C[1] - A[1]) * (p[0] - C[0]) + (A[0] - C[0]) * (p[1] - C[1])) / detT
        lambda3 = 1 - lambda1 - lambda2

        return 0 <= lambda1 <= 1 and 0 <= lambda2 <= 1 and 0 <= lambda3 <= 1

    def in_circle(self, a, b, c, p):
        # Calculate the determinant
        mat = np.array([
            [a[0] - p[0], a[1] - p[1], (a[0] - p[0])**2 + (a[1] - p[1])**2],
            [b[0] - p[0], b[1] - p[1], (b[0] - p[0])**2 + (b[1] - p[1])**2],
            [c[0] - p[0], c[1] - p[1], (c[0] - p[0])**2 + (c[1] - p[1])**2]
        ])
        det = np.linalg.det(mat)
        # Adjust the sign based on the orientation
        orientation = self.triangle_orientation(a, b, c)
        if orientation > 0:
            return det > 0
        else:
            return det < 0

    def plot(self, pause_time=0.1, show_current_point=True, show_circumcircles=True):
        plt.clf()
        ax = plt.gca()
        for a, b, c in self.triangles:
            triangle = np.array([self.vertices[a], self.vertices[b], self.vertices[c]])
            polygon = Polygon(triangle, edgecolor='blue', fill=None, linewidth=1)
            ax.add_patch(polygon)

        # Plot the current point if it exists and if show_current_point is True
        if show_current_point and self.current_point_index is not None:
            current_point = self.vertices[self.current_point_index]
            plt.scatter(current_point[0], current_point[1], color='red', zorder=5)

        # Plot the circumcircles for the current bad triangles if any and if show_circumcircles is True
        if show_circumcircles and self.current_bad_triangles:
            for a, b, c in self.current_bad_triangles:
                center, radius = self.circumcircle(self.vertices[a], self.vertices[b], self.vertices[c])
                if center is not None:
                    circle = Circle(center, radius, edgecolor='green', fill=False, linestyle='dotted')
                    ax.add_patch(circle)
                    # Optional: Draw a line from the center to the point to show inclusion
                    current_point = self.vertices[self.current_point_index]
                    ax.plot([center[0], current_point[0]], [center[1], current_point[1]], color='gray', linestyle='--')

        plt.xlim(-500, 500)
        plt.ylim(-500, 500)
        input()
        plt.pause(pause_time)

    def circumcircle(self, a, b, c):
        # Convert points to float for precision
        a = a.astype(float)
        b = b.astype(float)
        c = c.astype(float)

        # Calculate the determinant
        d = 2 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        if d == 0:
            return None, None  # Points are colinear

        # Calculate circumcenter coordinates
        ux = ((np.dot(a, a)*(b[1]-c[1]) + np.dot(b, b)*(c[1]-a[1]) + np.dot(c, c)*(a[1]-b[1])))/d
        uy = ((np.dot(a, a)*(c[0]-b[0]) + np.dot(b, b)*(a[0]-c[0]) + np.dot(c, c)*(b[0]-a[0])))/d
        center = np.array([ux, uy])

        # Calculate radius
        radius = np.linalg.norm(center - a)
        return center, radius

def main():
    np.random.seed(42)
    points = np.random.rand(25, 2) * 300 - 150
    triangulation = Triangulation()
    plt.figure(figsize=(10, 10))
    for point in points:
        # Check if the point is outside the super triangle
        if not triangulation.is_within_super_triangle(point):
            print(f"Point {point} is outside the super triangle and will be skipped.")
            # Reset state variables
            triangulation.current_point_index = None
            triangulation.current_bad_triangles = []
            # Call plot to keep the GUI responsive
            triangulation.plot(pause_time=0.5)
            continue
        triangulation.add_point(point)
    # Plot the final triangulation without red points and circles
    triangulation.current_point_index = None
    triangulation.current_bad_triangles = []
    input()
    triangulation.plot(pause_time=0, show_current_point=False, show_circumcircles=False)
    plt.show()

if __name__ == "__main__":
    main()
