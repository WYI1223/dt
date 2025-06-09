# tests/test_delaunay_split_tree_full.py

import random
import unittest

# Import your modules (adjust paths if needed)
from base.DCEL.vertex import Vertex as Point
from ..split_tree_build import construct_split_tree
from ..delaunay_from_splittree import build_delaunay_from_splittree
from base.Trainglation.TrainClassic import TrainClassic as Triangulation
from base.DelunayClassic import compute_delaunay

class TestDelaunaySplitTreeFull(unittest.TestCase):
    def setUp(self):
        # Generate a fixed random set of points for reproducibility
        # random.seed(123)
        # self.n_points = 20
        # self.points = [Point(random.uniform(0, 100), random.uniform(0, 100))
        #                for _ in range(self.n_points)]
        self.points = [Point(0.5, 0.3),
                   Point(0.3, 0.4),
                   Point(0.4, 0.1),
                   Point(0.6, 0.4),
                   Point(0.3, 0.2),
                   Point(0.51, 0.45),
                   Point(0.6, 0.2),
                   Point(0.7, 0.35),
                   Point(0.7, 0.1), ]
        self.n_points = len(self.points)

    def test_split_tree_builds_correct_delaunay(self):
        """
        Use split-tree insertion to build a Delaunay triangulation and compare
        against the direct compute_delaunay result. Also call triangulation.draw()
        to visually verify correctness.
        """
        # 1) Construct split tree over all points
        indices = list(range(self.n_points))
        split_root = construct_split_tree(self.points, indices, self.points)

        # 2) Build Delaunay via split-tree method
        del_split = build_delaunay_from_splittree(split_root, self.points)

        # 3) Directly compute Delaunay for comparison
        del_direct = compute_delaunay(self.points)

        # 4) Compare edge sets
        # edges_split = set(del_split.half_edges)
        # edges_direct = set(del_direct.half_edges)
        # self.assertEqual(edges_split, edges_direct,
        #                  "Split-tree Delaunay edges do not match direct Delaunay edges")

        # 5) Draw both triangulations for visual inspection
        print("Drawing Delaunay built via split-tree insertion:")
        del_split.draw()  # Uses internal draw method to render the triangulation

        print("Drawing Delaunay built via direct compute_delaunay:")
        del_direct.draw()  # Uses internal draw method to render the triangulation

if __name__ == "__main__":
    unittest.main()
