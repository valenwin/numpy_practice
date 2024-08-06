import unittest
import numpy as np
import os
from tasks.array_data import DataHandler


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.analyzer = DataHandler()

    def test_initial_array(self):
        self.assertEqual(self.analyzer.array.shape, (10, 10))

    def test_save_and_load_array(self):
        filename_base = "test_array"
        self.analyzer.save_array(filename_base)

        txt_array = self.analyzer.load_array(f"{filename_base}.txt", 'txt')
        csv_array = self.analyzer.load_array(f"{filename_base}.csv", 'csv')
        npy_array = self.analyzer.load_array(f"{filename_base}.npy", 'npy')

        np.testing.assert_array_almost_equal(self.analyzer.array, txt_array)
        np.testing.assert_array_almost_equal(self.analyzer.array, csv_array)
        np.testing.assert_array_equal(self.analyzer.array, npy_array)

        for ext in ['txt', 'csv', 'npy']:
            os.remove(f"{filename_base}.{ext}")

    def test_sum_array(self):
        self.assertEqual(self.analyzer.sum_array(), np.sum(self.analyzer.array))

    def test_mean_array(self):
        self.assertAlmostEqual(self.analyzer.mean_array(), np.mean(self.analyzer.array))

    def test_median_array(self):
        self.assertEqual(self.analyzer.median_array(), np.median(self.analyzer.array))

    def test_std_array(self):
        self.assertAlmostEqual(self.analyzer.std_array(), np.std(self.analyzer.array))

    def test_axis_aggregates(self):
        row_aggregates = self.analyzer.axis_aggregates(axis=1)
        self.assertEqual(row_aggregates['sum'].shape, (10,))
        self.assertEqual(row_aggregates['mean'].shape, (10,))
        self.assertEqual(row_aggregates['median'].shape, (10,))
        self.assertEqual(row_aggregates['std'].shape, (10,))

        col_aggregates = self.analyzer.axis_aggregates(axis=0)
        self.assertEqual(col_aggregates['sum'].shape, (10,))
        self.assertEqual(col_aggregates['mean'].shape, (10,))
        self.assertEqual(col_aggregates['median'].shape, (10,))
        self.assertEqual(col_aggregates['std'].shape, (10,))


if __name__ == '__main__':
    unittest.main()