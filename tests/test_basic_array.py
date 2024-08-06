import unittest
import numpy as np
from tasks.basic_array import BasicArrayManipulator


class TestArrayManipulator(unittest.TestCase):

    def setUp(self):
        self.manipulator = BasicArrayManipulator()

    def test_initial_arrays(self):
        self.assertEqual(self.manipulator.one_dim_array.shape, (10,))
        self.assertEqual(self.manipulator.two_dim_array.shape, (3, 3))
        np.testing.assert_array_equal(self.manipulator.one_dim_array, np.arange(1, 11))
        np.testing.assert_array_equal(self.manipulator.two_dim_array, np.arange(1, 10).reshape(3, 3))

    def test_get_third_element(self):
        self.assertEqual(self.manipulator.get_third_element(), 3)

    def test_get_first_two_rows_cols (self):
        expected = np.array([[1, 2], [4, 5]])
        np.testing.assert_array_equal(self.manipulator.get_first_two_rows_cols(), expected)

    def test_add_five_to_one_dim(self):
        expected = np.arange(6, 16)
        np.testing.assert_array_equal(self.manipulator.add_five_to_one_dim(), expected)

    def test_multiply_two_dim_by_two(self):
        expected = np.arange(2, 19, 2).reshape(3, 3)
        np.testing.assert_array_equal(self.manipulator.multiply_two_dim_by_two(), expected)


if __name__ == '__main__':
    unittest.main()
