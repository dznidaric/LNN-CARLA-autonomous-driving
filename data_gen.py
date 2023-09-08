"""
Generate batches of data.

Program that implements the DataGenerator Sequence class used for data generation.
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Class that instantiates the Keras data generators.
    """

    def __init__(self, x_files, y_files, dims=(80, 80), n_channels=3):
        """
        Initialize the data generator instance.

        :param x_files: input file names
        :param y_files: output (label, target) file names
        :param dims: image data dimensions (height x width)
        :param n_channels: image data channels (default 3 (RGB))
        """

        self.dims = dims
        self.x_files = x_files
        self.y_files = y_files
        self.n_batches = len(self)  # Number of batches
        self.n_channels = n_channels

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, index):
        # Load the input image frames at the current index
        x = np.load(self.x_files[index])

        # Transpose into human interpretable image (not necessary)
        x = x.transpose([0, 2, 3, 1])

        # Load the output steering angles at the current index
        y = np.load(self.y_files[index])

        return x, y

    def on_epoch_end(self):
        pass

    def __data_generation(self):
        pass

