import unittest
import numpy as np
from tasks.array_advanced import ArrayAdvanced


class TestArrayAdvanced(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.manipulator = ArrayAdvanced()

    def test_initial_array(self):
        self.assertEqual(self.manipulator.array.shape, (6, 6))

    def test_transpose(self):
        transposed = self.manipulator.transpose()
        np.testing.assert_array_equal(transposed, self.manipulator.array.T)

    def test_reshape(self):
        reshaped = self.manipulator.reshape((3, 12))
        self.assertEqual(reshaped.shape, (3, 12))
        np.testing.assert_array_equal(reshaped.flatten(), self.manipulator.array.flatten())

    def test_split(self):
        split_arrays = self.manipulator.split(3)
        self.assertEqual(len(split_arrays), 3)
        for arr in split_arrays:
            self.assertEqual(arr.shape, (2, 6))

    def test_combine(self):
        split_arrays = self.manipulator.split(3)
        combined = self.manipulator.combine(split_arrays)
        np.testing.assert_array_equal(combined, self.manipulator.array)


if __name__ == '__main__':
    unittest.main()
