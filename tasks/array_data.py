import numpy as np


class DataHandler:
    def __init__(self, shape=(10, 10)):
        np.random.seed(42)
        self.array = np.random.randint(1, 100, shape)

    @staticmethod
    def print_array(arr, message=None):
        if message:
            print(message)
        print(arr)
        print()

    @staticmethod
    def load_array(filename, file_type):
        if file_type == 'txt':
            return np.loadtxt(filename)
        elif file_type == 'csv':
            return np.loadtxt(filename, delimiter=',')
        elif file_type == 'npy':
            return np.load(filename)
        else:
            raise ValueError("Unsupported file type")

    def save_array(self, filename_base):
        np.savetxt(f"{filename_base}.txt", self.array)
        np.savetxt(f"{filename_base}.csv", self.array, delimiter=',')
        np.save(f"{filename_base}.npy", self.array)

    def sum_array(self):
        return np.sum(self.array)

    def mean_array(self):
        return np.mean(self.array)

    def median_array(self):
        return np.median(self.array)

    def std_array(self):
        return np.std(self.array)

    def axis_aggregates(self, axis):
        return {
            'sum': np.sum(self.array, axis=axis),
            'mean': np.mean(self.array, axis=axis),
            'median': np.median(self.array, axis=axis),
            'std': np.std(self.array, axis=axis)
        }


def main():
    data = DataHandler()

    data.print_array(data.array, "Initial 10x10 array:")

    data.save_array("data_array")

    txt_array = data.load_array("data_array.txt", 'txt')
    csv_array = data.load_array("data_array.csv", 'csv')
    npy_array = data.load_array("data_array.npy", 'npy')

    print("Loaded arrays:")
    data.print_array(txt_array, "From TXT:")
    data.print_array(csv_array, "From CSV:")
    data.print_array(npy_array, "From NPY:")

    assert np.array_equal(data.array, txt_array), "TXT loaded array doesn't match original"
    assert np.array_equal(data.array, csv_array), "CSV loaded array doesn't match original"
    assert np.array_equal(data.array, npy_array), "NPY loaded array doesn't match original"
    print("All loaded arrays match the original array.")

    print(f"Sum of array: {data.sum_array()}")
    print(f"Mean of array: {data.mean_array()}")
    print(f"Median of array: {data.median_array()}")
    print(f"Standard deviation of array: {data.std_array()}")

    print("\nRow-wise aggregates:")
    row_aggregates = data.axis_aggregates(axis=1)
    for key, value in row_aggregates.items():
        print(f"{key}: {value}")

    print("\nColumn-wise aggregates:")
    col_aggregates = data.axis_aggregates(axis=0)
    for key, value in col_aggregates.items():
        print(f"{key}: {value}")

    assert data.array.shape == (10, 10), "Array shape should be (10, 10)"
    assert np.isclose(data.mean_array(), np.mean(data.array)), "Mean calculation should match NumPy's mean"
    assert np.isclose(data.std_array(),
                      np.std(data.array)), "Standard deviation calculation should match NumPy's std"


if __name__ == "__main__":
    main()
