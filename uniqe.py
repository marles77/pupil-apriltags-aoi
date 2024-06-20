import numpy as np

# Example NumPy array
data = np.array([
    [1, 2, 3],
    [1, 5, 6],
    [1, 7, 8],
    [4, 10, 11],
    [4, 12, 13]
])

# Index column (assuming it's the first column, i.e., index 0)
index_column = 0

# Get unique indices and their first occurrence positions
unique_indices, unique_pos = np.unique(data[:, index_column], return_index=True)

# Select the rows corresponding to the first occurrence of each unique index
compressed_data = data[unique_pos]

print("Original data:\n", data)
print("Compressed data:\n", compressed_data)