import unittest
import numpy as np
from datetime import datetime
from tasks.ecommerce import ECommerceTransactions


class TestECommerceTransactions(unittest.TestCase):
    def setUp(self):
        self.analyzer = ECommerceTransactions()

    def test_initial_array_shape(self):
        self.assertEqual(self.analyzer.transactions.shape, (1000, 6))

    def test_total_revenue(self):
        revenue = self.analyzer.total_revenue()
        self.assertIsInstance(revenue, (int, float))
        self.assertGreater(revenue, 0)

    def test_unique_users(self):
        unique_users = self.analyzer.unique_users()
        self.assertIsInstance(unique_users, int)
        self.assertLessEqual(unique_users, 100)  # As we generated 100 unique user IDs

    def test_most_purchased_product(self):
        product = self.analyzer.most_purchased_product()
        self.assertIsInstance(product, (int, np.int64))
        self.assertGreaterEqual(product, 1)
        self.assertLess(product, 501)  # As we generated product IDs from 1 to 500

    def test_check_data_types(self):
        dtype = self.analyzer.check_data_types()
        self.assertIsInstance(dtype, np.dtype)

    def test_product_quantity_array(self):
        pq_array = self.analyzer.product_quantity_array()
        self.assertEqual(pq_array.shape, (1000, 2))

    def test_user_transaction_count(self):
        count = self.analyzer.user_transaction_count()
        self.assertEqual(len(count), 101)  # 100 users + 1 (for 0 index)
        self.assertEqual(count[0], 0)  # No user with ID 0

    def test_masked_array_zero_quantity(self):
        masked = self.analyzer.masked_array_zero_quantity()
        self.assertIsInstance(masked, np.ma.MaskedArray)
        self.assertTrue(np.all(masked.mask[:, 3] == (self.analyzer.transactions[:, 3] == 0)))
        self.assertTrue(np.all(masked.mask[:, 0] == masked.mask[:, 1]))

    def test_increase_prices(self):
        original_prices = self.analyzer.transactions[:, 4].copy()
        increased = self.analyzer.increase_prices(10)
        np.testing.assert_array_almost_equal(increased[:, 4], original_prices * 1.1)

    def test_filter_transactions(self):
        filtered = self.analyzer.filter_transactions()
        self.assertTrue(np.all(filtered[:, 3] > 1))

    def test_revenue_comparison(self):
        mid_year = int(datetime(2023, 7, 1).timestamp())
        end_year = int(datetime(2024, 1, 1).timestamp())
        rev1, rev2 = self.analyzer.revenue_comparison(mid_year, end_year)
        self.assertIsInstance(rev1, (int, float))
        self.assertIsInstance(rev2, (int, float))

    def test_user_transactions(self):
        user_trans = self.analyzer.user_transactions(1)
        self.assertTrue(np.all(user_trans[:, 1] == 1))

    def test_date_range_transactions(self):
        transactions = self.analyzer.date_range_transactions("2023-06-01", "2023-07-01")
        start_timestamp = int(datetime(2023, 6, 1).timestamp())
        end_timestamp = int(datetime(2023, 7, 1).timestamp())
        self.assertTrue(np.all((transactions[:, 5] >= start_timestamp) & (transactions[:, 5] < end_timestamp)))

    def test_top_products(self):
        top_prod_trans = self.analyzer.top_products()
        unique_products = np.unique(top_prod_trans[:, 2])
        self.assertLessEqual(len(unique_products), 5)

    def test_get_readable_dates(self):
        dates = self.analyzer.get_readable_dates()
        self.assertEqual(len(dates), 1000)
        for date in dates:
            self.assertTrue(date.startswith('2023-'))
            datetime.strptime(date, '%Y-%m-%d')  # This will raise ValueError if format is incorrect


if __name__ == '__main__':
    unittest.main()
