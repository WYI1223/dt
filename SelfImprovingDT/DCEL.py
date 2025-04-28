import numpy as np
import math
import matplotlib.pyplot as plt


# ---------- DCEL Data Structures ----------

class DCELVertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.incident_edge = None  # One incident half-edge

    def coord(self):
        return (self.x, self.y)


class DCELHalfEdge:
    def __init__(self, origin=None):
        self.origin = origin  # DCELVertex
        self.twin = None  # DCELHalfEdge
        self.next = None  # DCELHalfEdge
        self.prev = None  # DCELHalfEdge
        self.face = None  # DCELFace


class DCELFace:
    def __init__(self, edge=None):
        self.edge = edge  # One half-edge bounding this face
        # Cache circumcircle info for the triangle (face)
        self.circumcenter = None
        self.circumradius = None

    def compute_circumcircle(self):
        # Assume the face is a triangle.
        a = self.edge.origin.coord()
        b = self.edge.next.origin.coord()
        c = self.edge.next.next.origin.coord()
        (x1, y1), (x2, y2), (x3, y3) = a, b, c
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if d == 0:
            self.circumcenter = (0, 0)
            self.circumradius = float('inf')
        else:
            ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (
                        y1 - y2)) / d
            uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (
                        x2 - x1)) / d
            self.circumcenter = (ux, uy)
            self.circumradius = math.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
        return self.circumcenter, self.circumradius

    def circumcircle_contains(self, p):
        # Ensure the circumcircle is computed.
        if self.circumcenter is None or self.circumradius is None:
            self.compute_circumcircle()
        cx, cy = self.circumcenter
        dx, dy = p[0] - cx, p[1] - cy
        return math.sqrt(dx * dx + dy * dy) < self.circumradius


class DCEL:
    def __init__(self):
        self.vertices = []
        self.half_edges = []
        self.faces = []


# ---------- Helper Functions ----------

def oriented_area(a, b, c):
    """Returns twice the signed area of triangle abc.
       Positive if a, b, c are in counterclockwise order.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_in_face(face, p):
    """Determines whether point p lies inside the triangular face."""
    a = face.edge.origin.coord()
    b = face.edge.next.origin.coord()
    c = face.edge.next.next.origin.coord()
    area = oriented_area(a, b, c)
    area1 = oriented_area(p, b, c)
    area2 = oriented_area(a, p, c)
    area3 = oriented_area(a, b, p)
    # Adjust for orientation.
    if area < 0:
        area, area1, area2, area3 = -area, -area1, -area2, -area3
    return (area1 >= 0) and (area2 >= 0) and (area3 >= 0)


# ---------- Delaunay Triangulation with DCEL ----------

class DelaunayDCEL:
    def __init__(self, points):
        """
        Initializes the DCEL with a super-triangle and then inserts points incrementally.
        'points' is a list of (x, y) tuples.
        """
        self.dcel = DCEL()
        self.points = points
        self.initialize_super_triangle(points)
        # Insert points one-by-one (excluding those from the super-triangle).
        for p in points:
            self.insert_point(p)
        # (Optional) Remove faces adjacent to super-triangle vertices if desired.

    def initialize_super_triangle(self, points):
        """Creates a super-triangle large enough to contain all input points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = maxx - minx
        dy = maxy - miny
        deltaMax = max(dx, dy)
        midx = (minx + maxx) / 2
        midy = (miny + maxy) / 2
        # Create three vertices for the super-triangle.
        p1 = DCELVertex(midx - 20 * deltaMax, midy - deltaMax)
        p2 = DCELVertex(midx, midy + 20 * deltaMax)
        p3 = DCELVertex(midx + 20 * deltaMax, midy - deltaMax)
        self.dcel.vertices.extend([p1, p2, p3])
        # Create three half-edges (directed).
        e1 = DCELHalfEdge(p1)
        e2 = DCELHalfEdge(p2)
        e3 = DCELHalfEdge(p3)
        # Set up cycle pointers.
        e1.next = e2;
        e2.next = e3;
        e3.next = e1
        e1.prev = e3;
        e2.prev = e1;
        e3.prev = e2
        # For the super-triangle, twins remain None (they will be created as new faces share edges).
        # Create a face and assign the half-edges to it.
        face = DCELFace(e1)
        e1.face = face;
        e2.face = face;
        e3.face = face
        self.dcel.half_edges.extend([e1, e2, e3])
        self.dcel.faces.append(face)
        # Keep track of super-triangle vertices for later cleanup.
        self.super_triangle_vertices = [p1, p2, p3]

    def locate_face(self, p):
        """
        Naively locate a face (triangle) in the DCEL that contains point p.
        (A more advanced implementation would use an adaptive point-location structure.)
        """
        for face in self.dcel.faces:
            if point_in_face(face, p):
                return face
        return None  # Should not happen if p is within the super-triangle.

    def insert_point(self, p):
        """
        Insert a new point into the triangulation.
        Implements a simplified Bowyer-Watson update using the DCEL.
        """
        # Create new vertex.
        new_vertex = DCELVertex(p[0], p[1])
        self.dcel.vertices.append(new_vertex)
        # Step 1: Find all faces whose circumcircles contain p (the cavity).
        cavity = []
        for face in self.dcel.faces:
            if face.circumcircle_contains(p):
                cavity.append(face)
        if not cavity:
            # p lies outside the current triangulation (should not happen with a proper super-triangle)
            return

        # Step 2: Find boundary edges: these are half-edges whose twin face is not in the cavity.
        boundary_edges = []
        for face in cavity:
            current = face.edge
            for _ in range(3):
                twin_face = current.twin.face if current.twin and current.twin.face in self.dcel.faces else None
                if twin_face not in cavity:
                    boundary_edges.append(current)
                current = current.next

        # Step 3: Remove faces in the cavity.
        for face in cavity:
            if face in self.dcel.faces:
                self.dcel.faces.remove(face)
            # (A full implementation would also remove the half-edges that become interior.)

        # Step 4: For each boundary edge, create a new face connecting the new vertex with the edge.
        new_faces = []
        new_half_edges = []
        for be in boundary_edges:
            # The boundary edge be is shared between a face in the cavity and an external face.
            # Its endpoints are be.origin and be.next.origin.
            a = be.origin
            b = be.next.origin

            # Create three new half-edges for the new triangle:
            he1 = DCELHalfEdge(a)  # from a to new_vertex
            he2 = DCELHalfEdge(new_vertex)  # from new_vertex to b
            he3 = DCELHalfEdge(b)  # from b to a (will become the edge on the new face)

            # Set cycle pointers so that the new triangle has vertices (a, new_vertex, b).
            he1.next = he2;
            he2.next = he3;
            he3.next = he1
            he1.prev = he3;
            he2.prev = he1;
            he3.prev = he2

            # Create the new face.
            new_face = DCELFace(he1)
            he1.face = new_face;
            he2.face = new_face;
            he3.face = new_face
            new_faces.append(new_face)
            new_half_edges.extend([he1, he2, he3])
            self.dcel.faces.append(new_face)

        # Step 5: Add the new half-edges to the DCEL.
        self.dcel.half_edges.extend(new_half_edges)

        # Step 6: Update twin pointers among new half-edges.
        # (A robust implementation would use a hash table keyed by edge endpoints to find twins.)
        for he in new_half_edges:
            for other in new_half_edges:
                if he is not other:
                    # If he goes from u to v and other goes from v to u, set them as twins.
                    if he.origin == other.next.origin and other.origin == he.next.origin:
                        he.twin = other
                        other.twin = he


# ---------- Visualization Helper ----------

def plot_dcel(dcel, title="DCEL-based Delaunay Triangulation"):
    plt.figure(figsize=(8, 8))
    # Plot vertices.
    pts = np.array([(v.x, v.y) for v in dcel.vertices])
    plt.scatter(pts[:, 0], pts[:, 1], color="blue", s=10)
    # Plot each face (triangle).
    for face in dcel.faces:
        a = face.edge.origin.coord()
        b = face.edge.next.origin.coord()
        c = face.edge.next.next.origin.coord()
        triangle = np.array([a, b, c, a])
        plt.plot(triangle[:, 0], triangle[:, 1], "r-")
    plt.title(title)
    plt.axis("equal")
    plt.show()


# ---------- Example Usage of DCEL-based Delaunay ----------

if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic training data.
    cluster1 = np.random.randn(1, 2) + np.array([5, 5])
    cluster2 = np.random.randn(1, 2) + np.array([-5, -5])
    training_points = np.concatenate((cluster1, cluster2), axis=0).tolist()

    # Build the Delaunay triangulation using DCEL.
    delaunay_dcel = DelaunayDCEL(training_points)

    # Visualize the resulting triangulation.
    plot_dcel(delaunay_dcel.dcel, title="DCEL-based Delaunay Triangulation")
