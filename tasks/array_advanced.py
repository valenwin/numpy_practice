import numpy as np


class ArrayAdvanced:
    def __init__(self, shape=(6, 6)):
        np.random.seed(42)
        self.array = np.random.randint(1, 100, shape)

    @staticmethod
    def print_array(arr, message=None):
        if message:
            print(message)
        print(arr)
        print()

    def transpose(self):
        return np.transpose(self.array)

    def reshape(self, new_shape):
        return np.reshape(self.array, new_shape)

    def split(self, num_splits, axis=0):
        return np.split(self.array, num_splits, axis)

    def combine(self, arrays, axis=0):
        return np.concatenate(arrays, axis)


def main():
    arr_advanced = ArrayAdvanced()

    arr_advanced.print_array(arr_advanced.array, "Initial 6x6 array:")

    transposed = arr_advanced.transpose()
    arr_advanced.print_array(transposed, "Transposed array:")

    reshaped = arr_advanced.reshape((3, 12))
    arr_advanced.print_array(reshaped, "Reshaped array (3x12):")

    split_arrays = arr_advanced.split(3)
    for i, arr in enumerate(split_arrays):
        arr_advanced.print_array(arr, f"Split array {i + 1}:")

    combined = arr_advanced.combine(split_arrays)
    arr_advanced.print_array(combined, "Combined array:")

    assert arr_advanced.array.shape == (6, 6), "Initial array shape should be (6, 6)"
    assert transposed.shape == (6, 6), "Transposed array shape should be (6, 6)"
    assert reshaped.shape == (3, 12), "Reshaped array shape should be (3, 12)"
    assert len(split_arrays) == 3, "Should have 3 split arrays"
    assert combined.shape == arr_advanced.array.shape, "Combined array should have the same shape as the initial array"


if __name__ == "__main__":
    main()
