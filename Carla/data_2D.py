import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import ujson


class CARLA_Data(tf.data.Dataset):
    def __init__(self, root, config, current_epoch, batch_size=10):
        self.seq_len = np.array(config.seq_len)
        assert config.img_seq_len == 1

        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)
        self.scale = np.array(config.scale)

        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.inv_augment_prob = np.array(config.inv_augment_prob)

        self.image_shape = (160, 704, 3)
        self.measurements_shape = (5,)

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
            random.shuffle(self.video_dir)

    def data_generator(self):
        first_image = cv2.imread(
            str(self.video_dir[0] / "rgb" / ("%04d.png" % 2)), cv2.IMREAD_COLOR
        )
        self.image_shape = first_image.shape

        batch_images = []
        batch_measurements = []

        count = 0

        for video_dir in self.video_dir:

            num_seq = len(os.listdir(video_dir / "rgb"))

            for frame in range(2, num_seq - 2):
                with open(
                    str(video_dir / "measurements" / ("%04d.json" % frame)),
                    "r",
                    encoding="utf-8",
                ) as f1:
                    measurement = ujson.load(f1)

                if measurement["steer"] < 0.10 or measurement["steer"] < -0.10:
                    if random.random() > 0.3:
                        continue

                image = cv2.imread(
                    str(video_dir / "rgb" / ("%04d.png" % frame)), cv2.IMREAD_COLOR
                )
                if image is None:
                    print(
                        "Error loading file: ",
                        str(video_dir / "rgb" / ("%04d.png" % frame), encoding="utf-8"),
                    )

                """ 
                    IMAGE
                """
                image = image / 255.0
                image = np.array(image, dtype=np.float32)
                # augmentation
                crop_shift = 0
                degree = 0
                do_augment = (
                    self.augment and random.random() > self.inv_augment_prob
                )  # if random number is less than or equal to inv_augment_prob (0.15)
                if do_augment:
                    degree = (random.random() * 2.0 - 1.0) * self.aug_max_rotation
                    crop_shift = (
                        degree / 60 * self.img_width / self.scale
                    )  # we scale first

                image = crop_image_cv2(
                    image, crop=self.img_resolution, crop_shift=crop_shift
                )

                # MEASUREMENTS
                steer = measurement["steer"]
                throttle = measurement["throttle"]
                if(measurement["brake"]):
                    brake = 1.0
                else:
                    brake = 0.0
                if(measurement["light_hazard"]):
                    light = 1.0
                else:
                    light = 0.0
                if(measurement["stop_sign_hazard"]):
                    stop = 1.0
                else:
                    stop = 0.0


                batch_images.append(image)  # Append the image to the batch_images list
                count += 1
                batch_measurements.append(np.array([steer, throttle, brake, light, stop]))
                if len(batch_images) == self.batch_size:
                    print(np.array(batch_images).shape)
                    yield tf.stack(np.array(batch_images)), tf.stack(batch_measurements)
                    batch_images = []
                    batch_measurements = []


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
                )
                + self.image_shape,
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(self.batch_size, 5), dtype=tf.float32
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
