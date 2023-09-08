import matplotlib.pyplot as plt
import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Class that instantiates the Keras data generators.
    """

    def __init__(self, x_files, y_files, dims=(80, 80), n_channels=3):

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

        # Load the output steering angles at the current index
        y = np.load(self.y_files[index])

        return x, y

    def on_epoch_end(self):
        pass

    def __data_generation(self):
        pass

