import numpy as np


class BasicArrayManipulator:
    def __init__(self):
        self.one_dim_array = np.arange(1, 11)
        self.two_dim_array = np.arange(1, 10).reshape(3, 3)

    @staticmethod
    def print_array(arr, message=None):
        if message:
            print(message)
        print(arr)
        print()

    def get_third_element(self):
        return self.one_dim_array[2]

    def get_first_two_rows_cols(self):
        return self.two_dim_array[:2, :2]

    def add_five_to_one_dim(self):
        return self.one_dim_array + 5

    def multiply_two_dim_by_two(self):
        return self.two_dim_array * 2


def main():
    manipulator = BasicArrayManipulator()
    BasicArrayManipulator.print_array(manipulator.one_dim_array, "One-dimensional array:")
    BasicArrayManipulator.print_array(manipulator.two_dim_array, "Two-dimensional array:")

    # Indexing and Slicing
    third_element = manipulator.get_third_element()
    print(f"Third element of one-dimensional array: {third_element}")

    first_two_rows_cols = manipulator.get_first_two_rows_cols()
    BasicArrayManipulator.print_array(first_two_rows_cols, "First two rows and columns of two-dimensional array:")

    # Basic Arithmetic
    one_dim_plus_five = manipulator.add_five_to_one_dim()
    BasicArrayManipulator.print_array(one_dim_plus_five, "One-dimensional array + 5:")

    two_dim_times_two = manipulator.multiply_two_dim_by_two()
    BasicArrayManipulator.print_array(two_dim_times_two, "Two-dimensional array * 2:")


if __name__ == "__main__":
    main()
