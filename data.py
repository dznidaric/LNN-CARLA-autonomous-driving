import numpy as np
import tensorflow as tf


class CARLA_Data(tf.data.Dataset):
    def __init__(self, root, config):
        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)
        
        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)