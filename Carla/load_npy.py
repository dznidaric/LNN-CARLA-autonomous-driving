import os
from glob import glob

import numpy as np
from constants import TRAIN_DATA_DIR

""" # Specify the path to the .npy file
file_path = "64_batched_data_town05/images/64_images_0.npy"

# Load the .npy file
loaded_data = np.load(file_path)

# Now, you can work with the loaded data as a NumPy array
print(loaded_data)
print(loaded_data.shape) """

current_dir = os.getcwd()
print("Current working directory:", current_dir)

# Specify the path to the directory containing .npy images
path = "./64_batched_data_town05/images/*.npy"
print("Searching in directory:", path)

matching_files = glob(TRAIN_DATA_DIR)
print(matching_files)

# Print the list of matching files or directories
img = np.load(matching_files[0][0])
print(img.shape)