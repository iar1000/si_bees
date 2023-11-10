import unittest
import numpy as np

from src import utils

class TestUtils(unittest.TestCase):

    def test_relative_position(self):
        test_cases = [
            [(1,2), (4,6), (3, 4), (-3, -4)],
            [(0,0), (4,6), (4,6), (-4,-6)],
            [(0,0), (-2,3), (-2,3), (2,-3)],
            [(2,2), (-1,1), (-3,-1), (3,1)],
        ]
        for t in test_cases:
            self.assertEqual(utils.get_relative_pos(t[0], t[1]), t[2], msg="compute relative pos of two points")
            self.assertEqual(utils.get_relative_pos(t[1], t[0]), t[3], msg="compute inverse relative pos of two points")
    
    def test_relative_moore_to_linear(self):
        for radius in [1, 2, 3]:
            nh_size = (2 * radius + 1)**2
            worker = np.zeros(nh_size)
            print(f"radius={radius}, nh_size={nh_size}")
            rel_coords = list()
            for y_rel in range(-radius, radius+1):
                for x_rel in range(-radius, radius+1):
                    rel_coords.append((x_rel, y_rel))
            self.assertEqual(len(rel_coords), worker.size)

            for rc in rel_coords:
                i = utils.relative_moore_to_linear(rc, radius)
                self.assertGreaterEqual(i, 0)
                worker[i] = 1
            self.assertTrue(np.all(worker == 1))