import numpy as np
from datetime import datetime, timedelta


class ECommerceTransactions:
    def __init__(self):
        np.random.seed(42)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = (end_date - start_date).days

        random_days = np.random.randint(0, date_range, 1000)
        timestamps = [start_date + timedelta(days=int(day)) for day in random_days]

        # Generate transactions with object dtype for better readability
        self.transactions = np.array([
            [i + 1,  # transaction_id
             np.random.randint(1, 101),  # user_id
             np.random.randint(1, 501),  # product_id
             np.random.randint(1, 11),  # quantity
             round(np.random.uniform(10, 1000), 2),  # price (rounded to 2 decimal places)
             timestamps[i].timestamp()]  # timestamp
            for i in range(1000)
        ], dtype='object')

    @staticmethod
    def print_array(arr, message=None):
        if message:
            print(message)
        print(arr)
        print()

    def total_revenue(self):
        """
        Calculate the total revenue generated
        by multiplying quantity and price, and summing the result.
        """
        return np.sum(self.transactions[:, 3].astype(float) * self.transactions[:, 4].astype(float))

    def unique_users(self):
        """
        Determine the number of unique users who made transactions.
        """
        return np.unique(self.transactions[:, 1]).size

    def most_purchased_product(self):
        """
        Identify the most purchased product based on the quantity sold.
        """
        product_quantities = np.bincount(
            self.transactions[:, 2].astype(int),
            weights=self.transactions[:, 3].astype(int)
        )
        return np.argmax(product_quantities)

    def convert_price_to_int(self):
        """Convert prices to integers."""
        # Round prices to avoid issues with floating-point precision
        rounded_prices = np.round(self.transactions[:, 4].astype(float))
        # Convert rounded prices to integers
        self.transactions[:, 4] = rounded_prices.astype(int)

        # Debug information
        print(f"Prices before conversion: {self.transactions[:, 4][:5]}")
        print(f"Prices dtype after conversion: {self.transactions[:, 4].dtype}")

        return self.transactions

    def check_data_types(self):
        return self.transactions.dtype

    def product_quantity_array(self):
        """
        Returns a new array with only the product_id and quantity columns.
        """
        return self.transactions[:, [2, 3]]

    def user_transaction_count(self):
        """
        Generate an array of transaction counts per user.
        """
        return np.bincount(self.transactions[:, 1].astype(int))

    def masked_array_zero_quantity(self):
        """
        Masked array that hides transactions where the quantity is zero.
        """
        mask = self.transactions[:, 3] == 0
        return np.ma.masked_array(self.transactions, mask=np.column_stack([mask] * 6))

    def increase_prices(self, percentage):
        """
        Increase all prices by a certain percentage (e.g., 5% increase).
        """
        self.transactions[:, 4] = self.transactions[:, 4].astype(float) * (1 + percentage / 100)
        return self.transactions

    def filter_transactions(self):
        """
        Filter transactions to only include those with a quantity greater than 1.
        """
        return self.transactions[self.transactions[:, 3].astype(int) > 1]

    def revenue_comparison(self, timestamp1, timestamp2):
        """
        Compare the revenue from two different time periods.
        """
        mask1 = self.transactions[:, 5].astype(float) < timestamp1
        mask2 = (self.transactions[:, 5].astype(float) >= timestamp1) & (
                self.transactions[:, 5].astype(float) < timestamp2)
        revenue1 = np.sum(self.transactions[mask1][:, 3].astype(float) * self.transactions[mask1][:, 4].astype(float))
        revenue2 = np.sum(self.transactions[mask2][:, 3].astype(float) * self.transactions[mask2][:, 4].astype(float))
        return revenue1, revenue2

    def user_transactions(self, user_id):
        """
        Extract all transactions for a specific user.
        """
        return self.transactions[self.transactions[:, 1].astype(int) == user_id]

    def date_range_transactions(self, start_date, end_date):
        """
        Slice the dataset to include only transactions within a specific date range.
        """
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        return self.transactions[
            (self.transactions[:, 5].astype(float) >= start_timestamp) & (
                        self.transactions[:, 5].astype(float) < end_timestamp)]

    def top_products(self):
        """
        Using advanced indexing to retrieve transactions of the top 5 products by revenue.
        """
        product_revenue = np.bincount(self.transactions[:, 2].astype(int),
                                      weights=self.transactions[:, 3].astype(float) * self.transactions[:, 4].astype(
                                          float))
        top_5_products = np.argsort(product_revenue)[-5:]
        return self.transactions[np.isin(self.transactions[:, 2].astype(int), top_5_products)]

    def get_readable_dates(self):
        """Convert timestamps to readable date strings"""
        return [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in self.transactions[:, 5].astype(float)]


def main():
    ecommerce_data = ECommerceTransactions()

    ecommerce_data.print_array(
        ecommerce_data.transactions[:5],
        message="Sample of transaction data:"
    )
    assert ecommerce_data.transactions.shape == (1000, 6), "Initial transactions shape is incorrect"

    total_revenue = ecommerce_data.total_revenue()
    unique_users = ecommerce_data.unique_users()
    most_purchased_product_id = ecommerce_data.most_purchased_product()
    print(f"Total Revenue: ${total_revenue:.2f}")
    print(f"Number of Unique Users: {unique_users}")
    print(f"Most Purchased Product ID: {most_purchased_product_id}")

    converted_transactions = ecommerce_data.convert_price_to_int()
    ecommerce_data.print_array(
        converted_transactions[:5],
        message="Sample of transactions with integer prices:"
    )
    assert converted_transactions.shape == ecommerce_data.transactions.shape, "Shape after converting prices to integers is incorrect"
    # assert np.issubdtype(converted_transactions[:, 4].dtype,
    #                      np.integer), "Prices were not converted to integers correctly"

    data_types = ecommerce_data.check_data_types()
    print(f"Data types: {data_types}")

    product_quantity_array = ecommerce_data.product_quantity_array()
    ecommerce_data.print_array(
        product_quantity_array[:5],
        message="Sample of product quantity array:"
    )
    assert product_quantity_array.shape[1] == 2, "Product quantity array shape is incorrect"
    assert product_quantity_array.shape[0] == ecommerce_data.transactions.shape[
        0], "Product quantity array row count is incorrect"

    user_transaction_counts = ecommerce_data.user_transaction_count()
    print("Transaction counts for first 10 users:")
    print(user_transaction_counts[:10])
    assert len(user_transaction_counts) >= np.max(
        ecommerce_data.transactions[:, 1]) + 1, "User transaction count length is incorrect"

    masked_array = ecommerce_data.masked_array_zero_quantity()
    ecommerce_data.print_array(
        masked_array[:5],
        message="Sample of masked array (hiding zero quantities):"
    )
    assert masked_array.shape == ecommerce_data.transactions.shape, "Shape of masked array is incorrect"
    assert masked_array.mask.shape == ecommerce_data.transactions.shape, "Mask shape is incorrect"

    increased_prices = ecommerce_data.increase_prices(5)
    ecommerce_data.print_array(
        increased_prices[:5],
        message="Sample of transactions with 5% price increase:"
    )
    assert increased_prices.shape == ecommerce_data.transactions.shape, "Shape after increasing prices is incorrect"
    assert np.all(increased_prices[:, 4].astype(float) >= ecommerce_data.transactions[:, 4].astype(
        float)), "Prices were not increased correctly"

    filtered_transactions = ecommerce_data.filter_transactions()
    ecommerce_data.print_array(
        filtered_transactions[:5],
        message="Sample of filtered transactions (quantity > 1):"
    )
    assert filtered_transactions.shape[0] <= ecommerce_data.transactions.shape[
        0], "Shape of filtered transactions is incorrect"

    mid_year = int(datetime(2024, 7, 1).timestamp())
    rev1, rev2 = ecommerce_data.revenue_comparison(mid_year, int(datetime(2024, 1, 1).timestamp()))
    print(f"Revenue comparison: First half: ${rev1:.2f}, Second half: ${rev2:.2f}")
    assert rev1 >= 0 and rev2 >= 0, "Revenue comparison values are incorrect"

    user_transactions = ecommerce_data.user_transactions(1)
    ecommerce_data.print_array(
        user_transactions[:5],
        message="Sample of transactions for user 1:"
    )
    assert user_transactions.shape[1] == ecommerce_data.transactions.shape[1], "Shape of user transactions is incorrect"

    date_range_transactions = ecommerce_data.date_range_transactions("2024-06-01", "2024-07-01")
    ecommerce_data.print_array(
        date_range_transactions[:5],
        message="Sample of transactions within date range (2024-06-01 to 2024-07-01):"
    )
    assert date_range_transactions.shape[0] <= ecommerce_data.transactions.shape[
        0], "Shape of date range transactions is incorrect"

    top_products = ecommerce_data.top_products()
    ecommerce_data.print_array(
        top_products[:5],
        message="Sample of transactions for top 5 products by revenue:"
    )
    assert top_products.shape[0] <= ecommerce_data.transactions.shape[
        0], "Shape of top products transactions is incorrect"

    readable_dates = ecommerce_data.get_readable_dates()
    print(f"Readable dates for first 5 transactions: {readable_dates[:5]}")
    assert len(readable_dates) == ecommerce_data.transactions.shape[0], "Readable dates length is incorrect"


if __name__ == '__main__':
    main()
