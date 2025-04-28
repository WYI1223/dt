import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict


# ---------- Geometry Helper Functions ----------

def oriented_area(a, b, c):
    """Returns twice the signed area of triangle abc.
       Positive if a, b, c are in counterclockwise order.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_in_triangle(tri, p):
    """Check if point p is inside triangle tri using barycentric coordinates.
       tri.vertices is a tuple (a, b, c).
    """
    a, b, c = tri.vertices
    # Compute oriented areas
    area = oriented_area(a, b, c)
    area1 = oriented_area(p, b, c)
    area2 = oriented_area(a, p, c)
    area3 = oriented_area(a, b, p)
    # Allow points on edges (>= 0)
    if area < 0:
        area, area1, area2, area3 = -area, -area1, -area2, -area3
    return (area1 >= 0) and (area2 >= 0) and (area3 >= 0)


def common_edge(tri1, tri2):
    """If triangles tri1 and tri2 share an edge, return that edge as a tuple of vertices (sorted).
       Otherwise, return None.
    """
    set1 = set(tri1.vertices)
    set2 = set(tri2.vertices)
    common = set1.intersection(set2)
    if len(common) == 2:
        return tuple(sorted(common))
    return None


# ---------- Triangle Class with Neighbor Pointers ----------

class Triangle:
    def __init__(self, vertices):
        """
        vertices: tuple of 3 points (each a tuple (x,y)) in counterclockwise order.
        neighbors: list of 3 neighboring triangles (None if no neighbor), where neighbor[i] is opposite vertices[i].
        """
        self.vertices = vertices
        self.neighbors = [None, None, None]
        self.circumcenter, self.circumradius = self.compute_circumcircle()

    def compute_circumcircle(self):
        (x1, y1), (x2, y2), (x3, y3) = self.vertices
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if d == 0:
            # Degenerate triangle: return a dummy circumcenter and infinite radius
            return ((0, 0), float('inf'))
        ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
        uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
        center = (ux, uy)
        radius = math.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
        return center, radius

    def circumcircle_contains(self, p):
        cx, cy = self.circumcenter
        dx, dy = p[0] - cx, p[1] - cy
        return math.sqrt(dx * dx + dy * dy) < self.circumradius


# ---------- Detailed Incremental Delaunay Triangulation ----------

class DetailedDelaunay:
    def __init__(self, points):
        """
        Build an initial Delaunay triangulation from a list of points.
        Uses a super-triangle and then inserts each point incrementally.
        """
        self.triangles = []
        self.last_triangle = None  # For "walk" based point location
        self.points = points

        # Compute a bounding box for the points.
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = maxx - minx
        dy = maxy - miny
        deltaMax = max(dx, dy)
        midx = (minx + maxx) / 2
        midy = (miny + maxy) / 2

        # Create a super-triangle that surely contains all points.
        p1 = (midx - 20 * deltaMax, midy - deltaMax)
        p2 = (midx, midy + 20 * deltaMax)
        p3 = (midx + 20 * deltaMax, midy - deltaMax)
        super_triangle = Triangle((p1, p2, p3))
        self.triangles.append(super_triangle)

        # Insert each point into the triangulation.
        for p in points:
            self.insert_point(p)

    def locate_triangle(self, p):
        """
        Locate a triangle that contains point p using a "walk" in the neighbor graph.
        Starts from the last found triangle.
        """
        if self.last_triangle is None:
            self.last_triangle = self.triangles[0]
        t = self.last_triangle
        # Try a fixed number of neighbor steps.
        for _ in range(100):
            if point_in_triangle(t, p):
                self.last_triangle = t
                return t
            # Walk to a neighbor that might contain p.
            found = False
            for nb in t.neighbors:
                if nb is not None and point_in_triangle(nb, p):
                    t = nb
                    found = True
                    break
            if not found:
                break  # No neighbor seems to contain p; break out.
        # Fallback: linear search.
        for tri in self.triangles:
            if point_in_triangle(tri, p):
                self.last_triangle = tri
                return tri
        return t  # Should rarely happen.

    def insert_point(self, p):
        """
        Insert point p into the triangulation using an incremental Bowyer–Watson approach.
        """
        # 1. Locate a triangle containing p.
        t0 = self.locate_triangle(p)

        # 2. Identify all triangles whose circumcircles contain p (the "cavity").
        bad_triangles = set()
        stack = [t0]
        while stack:
            t = stack.pop()
            if t in bad_triangles:
                continue
            if t.circumcircle_contains(p):
                bad_triangles.add(t)
                for nb in t.neighbors:
                    if nb is not None and nb not in bad_triangles:
                        stack.append(nb)

        # 3. Find the boundary polygon (edges shared by only one triangle in bad_triangles).
        boundary = []
        for tri in bad_triangles:
            for i in range(3):
                nb = tri.neighbors[i]
                # Edge opposite vertex i.
                edge = (tri.vertices[(i + 1) % 3], tri.vertices[(i + 2) % 3])
                if nb not in bad_triangles:
                    boundary.append(edge)

        # 4. Remove the bad triangles from the triangulation.
        for tri in bad_triangles:
            if tri in self.triangles:
                self.triangles.remove(tri)

        # 5. Create new triangles from each boundary edge to the new point.
        new_triangles = []
        for edge in boundary:
            a, b = edge
            # Ensure edge is in counterclockwise order with respect to p.
            if oriented_area(a, b, p) < 0:
                a, b = b, a
            new_tri = Triangle((a, b, p))
            new_triangles.append(new_tri)

        # 6. Update neighbor pointers among new triangles.
        # For each pair of new triangles, if they share an edge, link them as neighbors.
        for i, t1 in enumerate(new_triangles):
            for j, t2 in enumerate(new_triangles):
                if i == j:
                    continue
                edge = common_edge(t1, t2)
                if edge is not None:
                    # Find which edge in t1 is the common one.
                    for k in range(3):
                        edge_t1 = (t1.vertices[(k + 1) % 3], t1.vertices[(k + 2) % 3])
                        if set(edge_t1) == set(edge):
                            t1.neighbors[k] = t2
        # 7. Add the new triangles into the global triangulation.
        self.triangles.extend(new_triangles)


# ---------- Self-Improving Delaunay Framework ----------

class SelfImprovingDelaunay:
    def __init__(self, backbone_fraction=0.2):
        """
        backbone_fraction: Fraction of training points to use as the ε-net backbone.
        """
        self.backbone_fraction = backbone_fraction
        self.backbone_points = []
        self.detailed_triangulation = None
        self.trained = False

    def training_phase(self, training_point_sets):
        """
        Aggregate training sets and choose a random subset as the backbone.
        Build the initial triangulation on these backbone points.
        training_point_sets: list of point sets (each is a list of (x, y) tuples).
        """
        all_points = []
        for pts in training_point_sets:
            all_points.extend(pts)
        all_points = np.array(all_points)
        num_backbone = max(3, int(len(all_points) * self.backbone_fraction))
        indices = np.random.choice(len(all_points), num_backbone, replace=False)
        self.backbone_points = [tuple(all_points[i]) for i in indices]
        # Build detailed triangulation on the backbone points.
        self.detailed_triangulation = DetailedDelaunay(self.backbone_points)
        self.trained = True
        print(f"Training completed with {len(self.backbone_points)} backbone points.")

    def limiting_phase(self, new_points):
        """
        Insert new points into the triangulation built during training.
        new_points: list of (x, y) tuples.
        Returns the updated list of triangles.
        """
        if not self.trained:
            raise Exception("Training phase not completed!")
        for p in new_points:
            self.detailed_triangulation.insert_point(tuple(p))
        return self.detailed_triangulation.triangles


# ---------- Visualization Helper ----------

def plot_detailed_triangulation(triangles, title="Detailed Delaunay Triangulation"):
    plt.figure(figsize=(8, 8))
    # Collect all unique points for scatter plotting.
    pts = []
    for tri in triangles:
        for p in tri.vertices:
            pts.append(p)
    pts = np.array(list(set(pts)))
    plt.scatter(pts[:, 0], pts[:, 1], color="blue", s=10)
    # Draw each triangle.
    for t in triangles:
        pts_tri = np.array(list(t.vertices) + [t.vertices[0]])
        plt.plot(pts_tri[:, 0], pts_tri[:, 1], "r-")
    plt.title(title)
    plt.axis("equal")
    plt.show()


# ---------- Example Usage ----------

if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic training data: two clusters.
    cluster1 = np.random.randn(50, 2) + np.array([5, 5])
    cluster2 = np.random.randn(50, 2) + np.array([-5, -5])
    training_sets = [cluster1.tolist(), cluster2.tolist()]

    # Create the self-improving Delaunay instance.
    sid = SelfImprovingDelaunay(backbone_fraction=0.2)
    sid.training_phase(training_sets)

    # Generate new points (from a similar distribution) for the limiting phase.
    new_points = (np.random.randn(30, 2) + np.array([0, 0])).tolist()
    updated_triangles = sid.limiting_phase(new_points)

    # Combine backbone and new points for visualization.
    all_backbone = sid.backbone_points
    all_new = [tuple(p) for p in new_points]
    plot_detailed_triangulation(updated_triangles, title="Self-Improving Delaunay Triangulation")
