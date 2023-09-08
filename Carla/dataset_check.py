import os
import random
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import ujson
from tqdm import tqdm


class CARLA_Data(tf.data.Dataset):
    def __init__(self, root, config, current_epoch, batch_size=8):
        self.seq_len = np.array(config.seq_len)
        assert config.img_seq_len == 1

        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)
        self.scale = np.array(config.scale)

        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.inv_augment_prob = np.array(config.inv_augment_prob)

        self.image_shape = (160, 704, 3)
        self.measurements_shape = (4,)

        self.batch_size = batch_size

        self.temporal_length = np.array(config.temporal_length)
        self.temporal_stride = np.array(config.temporal_stride)

        self.current_epoch = current_epoch

        self.video_dir = []

        for sub_root in root:
            sub_root = Path(sub_root)

            # list sub-directories in root
            root_files = os.listdir(sub_root)
            routes = [
                folder
                for folder in root_files
                if not os.path.isfile(os.path.join(sub_root, folder))
            ]
            for route in routes:
                route_dir = sub_root / route
                self.video_dir.append(route_dir)

    def data_generator(self):

        steps = 0
        batch_measurements = []
        i=0

        for video_dir in self.video_dir[self.current_epoch*10:(self.current_epoch+1)*10]:
            sample_measurements = []

            num_seq = len(os.listdir(video_dir / "measurements"))
            if num_seq - 4 < 16:
                continue
            for frame in range(2, num_seq - 2):

                sample_measurements.append(np.zeros(1))
                if len(sample_measurements) == self.temporal_length:

                    # add samples to batch
                    batch_measurements.append(sample_measurements)
                    print("len(batch_measurements): ", len(batch_measurements))
                    sample_measurements = []
                    if len(batch_measurements) == self.batch_size:
                        steps += 1
                        batch_measurements = []
            i += 1
            print("#DIR: ", i)
        print("steps: ", steps)

    def _inputs(self):
        # Return the structure of input elements
        return ()  # No inputs in this case

    @property
    def element_spec(self):
        # Return the structure of elements in the dataset
        return (
            tf.TensorSpec(
                shape=(
                    self.batch_size,
                    self.temporal_length,
                )
                + self.image_shape,
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(self.batch_size, self.temporal_length, 4), dtype=tf.float32
            )
        )

    def create_dataset(self):
        return tf.data.Dataset.from_generator(
            self.data_generator, output_signature=self.element_spec
        )


def crop_image(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.width
    height = image.height
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    image = np.asarray(image)
    cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    return cropped_image


def crop_image_cv2(image, crop=(128, 640), crop_shift=0):
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    return cropped_image
