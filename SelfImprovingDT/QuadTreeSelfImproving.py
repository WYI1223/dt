import numpy as np

class QuadTreeNode:
    def __init__(self, bounds, depth=0, max_depth=10):
        self.bounds = bounds  # (xmin, ymin, xmax, ymax)
        self.points = []
        self.children = []
        self.depth = depth
        self.max_depth = max_depth

    def insert(self, point):
        if self.children:
            # Insert into appropriate child
            self._get_child(point).insert(point)
        else:
            self.points.append(point)
            if len(self.points) > 4 and self.depth < self.max_depth:
                self.subdivide()

    def subdivide(self):
        xmin, ymin, xmax, ymax = self.bounds
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        self.children = [
            QuadTreeNode((xmin, ymin, xmid, ymid), self.depth + 1, self.max_depth),
            QuadTreeNode((xmid, ymin, xmax, ymid), self.depth + 1, self.max_depth),
            QuadTreeNode((xmin, ymid, xmid, ymax), self.depth + 1, self.max_depth),
            QuadTreeNode((xmid, ymid, xmax, ymax), self.depth + 1, self.max_depth),
        ]

        for point in self.points:
            self._get_child(point).insert(point)
        self.points = []

    def _get_child(self, point):
        x, y = point
        xmin, ymin, xmax, ymax = self.bounds
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        if x < xmid:
            if y < ymid:
                return self.children[0]
            else:
                return self.children[2]
        else:
            if y < ymid:
                return self.children[1]
            else:
                return self.children[3]

    def query(self, point):
        if self.children:
            return self._get_child(point).query(point)
        else:
            return self.points

class DelaunayTriangulation:
    def __init__(self):
        self.triangles = []
        self.super_triangle = None
        self.points = []
        self.quad_tree = None

    def circumcircle_contains(self, triangle, point):
        # Correctly compute if the point is inside the circumcircle of the triangle
        ax, ay = triangle[0]
        bx, by = triangle[1]
        cx, cy = triangle[2]
        dx, dy = point

        # Build the matrix
        mat = np.array([
            [ax - dx, ay - dy, (ax - dx)**2 + (ay - dy)**2],
            [bx - dx, by - dy, (bx - dx)**2 + (by - dy)**2],
            [cx - dx, cy - dy, (cx - dx)**2 + (cy - dy)**2]
        ])

        det = np.linalg.det(mat)

        # Orientation correction
        # Compute the orientation of triangle ABC
        orientation = (bx - ax)*(cy - ay) - (by - ay)*(cx - ax)

        if orientation > 0:
            # Counter-clockwise orientation
            return det > 0
        else:
            # Clockwise orientation
            return det < 0

    def add_super_triangle(self):
        # Create a super triangle that encompasses all the points
        xmin = min(self.points, key=lambda p: p[0])[0]
        xmax = max(self.points, key=lambda p: p[0])[0]
        ymin = min(self.points, key=lambda p: p[1])[1]
        ymax = max(self.points, key=lambda p: p[1])[1]

        dx = xmax - xmin
        dy = ymax - ymin
        delta_max = max(dx, dy) * 100

        midx = (xmin + xmax) / 2
        midy = (ymin + ymax) / 2

        p1 = (midx - delta_max, midy - delta_max)
        p2 = (midx, midy + delta_max)
        p3 = (midx + delta_max, midy - delta_max)

        self.super_triangle = (p1, p2, p3)
        self.triangles.append(self.super_triangle)

    def remove_super_triangle(self):
        # Remove triangles connected to the super triangle
        def is_super_triangle_vertex(vertex):
            return vertex in self.super_triangle

        self.triangles = [
            triangle for triangle in self.triangles
            if not any(is_super_triangle_vertex(vertex) for vertex in triangle)
        ]

    def insert_point(self, point):
        bad_triangles = []
        polygon = []

        # Find all triangles that are no longer valid due to the insertion
        for triangle in self.triangles:
            if self.circumcircle_contains(triangle, point):
                bad_triangles.append(triangle)

        # Find the boundary of the polygonal hole
        for triangle in bad_triangles:
            for edge in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0])
            ]:
                # Check if the edge is shared by only one triangle (it's on the boundary)
                shared = False
                for other in bad_triangles:
                    if other == triangle:
                        continue
                    if edge in [
                        (other[0], other[1]),
                        (other[1], other[2]),
                        (other[2], other[0]),
                        (other[1], other[0]),
                        (other[2], other[1]),
                        (other[0], other[2])
                    ]:
                        shared = True
                        break
                if not shared:
                    polygon.append(edge)

        # Remove the bad triangles
        for triangle in bad_triangles:
            self.triangles.remove(triangle)

        # Re-triangulate the polygonal hole
        for edge in polygon:
            new_triangle = (edge[0], edge[1], point)
            self.triangles.append(new_triangle)

    def triangulate(self):
        self.add_super_triangle()

        for point in self.points:
            self.insert_point(point)

        self.remove_super_triangle()

    def learning_phase(self, input_samples):
        # Build a quadtree for point location
        print("Learning phase: Building quadtree...")
        xmin = min(input_samples, key=lambda p: p[0])[0]
        xmax = max(input_samples, key=lambda p: p[0])[0]
        ymin = min(input_samples, key=lambda p: p[1])[1]
        ymax = max(input_samples, key=lambda p: p[1])[1]

        self.quad_tree = QuadTreeNode((xmin, ymin, xmax, ymax))

        for point in input_samples:
            self.quad_tree.insert(point)

        self.points.extend(input_samples)
        print("Learning phase completed.")

    def limiting_phase(self, new_points):
        # Use the quadtree for efficient point insertion
        print("Limiting phase: Inserting new points...")
        for point in new_points:
            # Efficient point location could be used here
            self.points.append(point)
            self.insert_point(point)
        print("Limiting phase completed.")

    def compute_delaunay(self, input_points):
        if not self.quad_tree:
            # If learning phase not done, perform it
            self.learning_phase(input_points)
            self.triangulate()
        else:
            self.limiting_phase(input_points)
            # After inserting new points, remove super triangle again
            self.remove_super_triangle()
        return self.triangles

# Example usage
if __name__ == "__main__":
    import time

    def timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print("{} 函数运行时间：{:.6f}秒".format(func.__name__, end_time - start_time))
            return result

        return wrapper

    # Generate sample data
    np.random.seed(42)
    learning_data = np.random.rand(100, 2).tolist()

    delaunay = DelaunayTriangulation()
    # Learning phase
    delaunay.learning_phase(learning_data)
    delaunay.triangulate()

    @timer
    def test():
        # Limiting phase with new data
        new_data = np.random.rand(100, 2).tolist()
        delaunay.limiting_phase(new_data)
        # Remove super triangle after inserting new points
        delaunay.remove_super_triangle()

        # Access the computed triangulation
        print("Computed triangles:")
        for triangle in delaunay.triangles:
            print(triangle)
    test()
