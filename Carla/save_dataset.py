"""
DataPreprocessor class implementation.
"""
import os
import random
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import ujson
from config import GlobalConfig
from tqdm import tqdm

config = GlobalConfig(setting="town_05")

root = config.data


img_width = np.array(config.img_width)
augment = np.array(config.augment)
aug_max_rotation = np.array(config.aug_max_rotation)
inv_augment_prob = np.array(config.inv_augment_prob)


def process_image(image):
    image = image / 255.0
    image = np.array(image, dtype=np.float32)

    # augmentation
    crop_shift = 0
    degree = 0
    do_augment = (
        augment and random.random() > inv_augment_prob
    )  # if random number is less than or equal to inv_augment_prob (0.15)
    if do_augment:
        degree = (random.random() * 2.0 - 1.0) * aug_max_rotation
        crop_shift = degree / 60 * img_width  # we scale first

    crop = (128, 640)
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    return image[start_y : start_y + crop_h, start_x : start_x + crop_w]


def npy_chunks(data, inner_folder, chunk_size=64, outer_folder="town05", chunk_start=0):
    file_name = f"{chunk_size}_{inner_folder}"
    path = f"{chunk_size}_batched_data_{outer_folder}/{inner_folder}"
    saved = 0
    for idx, i in tqdm(enumerate(range(0, data.shape[0], chunk_size))):
        chunk = data[i : i + chunk_size]

        if chunk.shape[0] < chunk_size:
            continue
        try:
            np.save(f"{path}/{file_name}_{idx + chunk_start}", chunk)
        except FileNotFoundError:
            os.makedirs(f"{path}")
            np.save(f"{path}/{file_name}_{idx + chunk_start}", chunk)
        
        saved = idx
    return saved + 1


def plot_data(data, title, x_label, y_label):
    plt.figure(figsize=(10, 8))
    plt.plot(data, color="b")  # peak at 13850
    plt.title(title, weight="bold", fontsize=16)
    plt.xlabel(x_label, weight="bold", fontsize=14)
    plt.ylabel(y_label, weight="bold", fontsize=14)
    plt.xticks(weight="bold", fontsize=12, rotation=45)
    plt.yticks(weight="bold", fontsize=12)
    plt.grid(color="y", linewidth=0.5)
    plt.show()



chunk_start = 0
for sub_root in root:
    sub_root = Path(sub_root)

    root_files = os.listdir(sub_root)
    routes = [
        folder
        for folder in root_files
        if not os.path.isfile(os.path.join(sub_root, folder))
    ]
    for route in tqdm(routes):
        sample_images = []
        steer_angles = []
       
        route_dir = sub_root / route

        for video_dir in route_dir.glob("**/rgb"):

            num_seq = len(os.listdir(video_dir))
            for frame in range(2, num_seq - 2):

                image = cv2.imread(
                    str(video_dir / ("%04d.png" % frame)), cv2.IMREAD_COLOR
                )
                if image is None:
                    print(
                        "Error loading file: ",
                        str(video_dir / ("%04d.png" % frame), encoding="utf-8"),
                    )

                image = process_image(image)
                sample_images.append(image)

        for video_dir in route_dir.glob("**/measurements"):

            num_seq = len(os.listdir(video_dir))
            for frame in range(2, num_seq - 2):
                with open(
                    str(video_dir / ("%04d.json" % frame)),
                    "r",
                    encoding="utf-8",
                ) as f1:
                    measurement = ujson.load(f1)
                steer = measurement["steer"]
                steer_angles.append(steer)

    # Save the shuffled and resized camera frames as indexed, batch-sized NumPy files
        npy_chunks(np.array(sample_images), "images", chunk_start=chunk_start)

        # Save the shuffled and scaled steering angle values as indexed, batch-sized NumPy files
        steer_angles = np.array(steer_angles)
        chunk_start += npy_chunks(steer_angles, "steering", chunk_start=chunk_start)
plot_data(steer_angles, "Steering angles", "Frame no.", "Steering angle (scaled)")
