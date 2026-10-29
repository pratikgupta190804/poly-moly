import numpy as np

# Load the file
data = np.load('features/fp_raw.npy', mmap_mode='r')

# Inspect the contents
print(data)