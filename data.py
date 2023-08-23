""" 
@article{Chitta2022PAMI,
  author = {Chitta, Kashyap and
            Prakash, Aditya and
            Jaeger, Bernhard and
            Yu, Zehao and
            Renz, Katrin and
            Geiger, Andreas},
  title = {TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving},
  journal = {Pattern Analysis and Machine Intelligence (PAMI)},
  year = {2022},
} 
@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and
            Chitta, Kashyap and
            Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
} """

""" github repo: https://github.com/autonomousvision/transfuser#evaluation """


import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import ujson
from tqdm import tqdm


class CARLA_Data(tf.data.Dataset):
    def __init__(self, root, config):
        self.seq_len = np.array(config.seq_len)
        assert config.img_seq_len == 1
        self.pred_len = np.array(config.pred_len)

        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)

        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)

        self.images = []
        self.labels = []
        self.measurements = []

        for sub_root in tqdm(root, file=sys.stdout):
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
                num_seq = len(os.listdir(route_dir / "lidar"))

                # ignore the first two and last two frame
                for seq in range(2, num_seq - self.pred_len - self.seq_len - 2):
                    # load input seq and pred seq jointly
                    image = []
                    label = []
                    measurement = []
                    # Loads the current (and past) frames (if seq_len > 1)
                    for idx in range(self.seq_len):
                        image.append(route_dir / "rgb" / ("%04d.png" % (seq + idx)))
                        measurement.append(
                            route_dir / "measurements" / ("%04d.json" % (seq + idx))
                        )

                    # Additionally load future labels of the waypoints
                    for idx in range(self.seq_len + self.pred_len):
                        label.append(
                            route_dir / "label_raw" / ("%04d.json" % (seq + idx))
                        )

                    self.images.append(image)
                    self.labels.append(label)
                    self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.images = np.array(self.images).astype(np.string_)
        self.labels = np.array(self.labels).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)

    def __len__(self):
        """Returns the length of the dataset."""
        return self.images.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx."""
        cv2.setNumThreads(
            0
        )  # Disable threading because the data loader will already split in threads.

        data = dict()

        images = self.images[index]
        labels = self.labels[index]
        measurements = self.measurements[index]

        # load measurements
        loaded_images = []
        loaded_labels = []
        loaded_measurements = []

        # Because the strings are stored as numpy byte objects we need to convert them back to utf-8 strings
        # Since we also load labels for future timesteps, we load and store them separately
        for i in range(self.seq_len + self.pred_len):
            if (not (self.data_cache is None)) and (
                str(labels[i], encoding="utf-8") in self.data_cache
            ):
                labels_i = self.data_cache[str(labels[i], encoding="utf-8")]
            else:
                with open(str(labels[i], encoding="utf-8"), "r") as f2:
                    labels_i = ujson.load(f2)

                if not self.data_cache is None:
                    self.data_cache[str(labels[i], encoding="utf-8")] = labels_i

            loaded_labels.append(labels_i)

        for i in range(self.seq_len):
            if (
                not self.data_cache is None
                and str(measurements[i], encoding="utf-8") in self.data_cache
            ):
                measurements_i, images_i = self.data_cache[
                    str(measurements[i], encoding="utf-8")
                ]
                images_i = cv2.imdecode(images_i, cv2.IMREAD_UNCHANGED)

            else:
                with open(str(measurements[i], encoding="utf-8"), "r") as f1:
                    measurements_i = ujson.load(f1)

                images_i = cv2.imread(
                    str(images[i], encoding="utf-8"), cv2.IMREAD_COLOR
                )
                if images_i is None:
                    print("Error loading file: ", str(images[i], encoding="utf-8"))
                images_i = scale_image_cv2(
                    cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB), self.scale
                )

                if not self.data_cache is None:
                    # We want to cache the images in png format instead of uncompressed, to reduce memory usage
                    result, compressed_imgage = cv2.imencode(".png", images_i)
                    self.data_cache[str(measurements[i], encoding="utf-8")] = (
                        measurements_i,
                        compressed_imgage,
                    )

            loaded_images.append(images_i)
            loaded_measurements.append(measurements_i)

        labels = loaded_labels
        measurements = loaded_measurements

        # load image, only use current frame
        # augment here
        crop_shift = 0
        degree = 0
        rad = np.deg2rad(degree)
        do_augment = self.augment and random.random() > self.inv_augment_prob
        if do_augment:
            degree = (random.random() * 2.0 - 1.0) * self.aug_max_rotation
            rad = np.deg2rad(degree)
            crop_shift = degree / 60 * self.img_width / self.scale  # we scale first

        images_i = loaded_images[self.seq_len - 1]
        images_i = crop_image_cv2(
            images_i, crop=self.img_resolution, crop_shift=crop_shift
        )

        data["rgb"] = images_i

        # ego car is always the first one in label file
        ego_id = labels[self.seq_len - 1][0]["id"]

        # only use label of frame 1
        bboxes = parse_labels(labels[self.seq_len - 1], rad=-rad)
        waypoints = get_waypoints(labels[self.seq_len - 1 :], self.pred_len + 1)
        waypoints = transform_waypoints(waypoints)

        # save waypoints in meters
        filtered_waypoints = []
        for id in list(bboxes.keys()) + [ego_id]:
            waypoint = []
            for matrix, flag in waypoints[id][1:]:
                waypoint.append(matrix[:2, 3])
            filtered_waypoints.append(waypoint)
        waypoints = np.array(filtered_waypoints)

        label = []
        for id in bboxes.keys():
            label.append(bboxes[id])
        label = np.array(label)

        # padding
        label_pad = np.zeros((20, 7), dtype=np.float32)
        ego_waypoint = waypoints[-1]

        # for the augmentation we only need to transform the waypoints for ego car
        degree_matrix = np.array(
            [[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]]
        )
        ego_waypoint = (degree_matrix @ ego_waypoint.T).T

        if label.shape[0] > 0:
            label_pad[: label.shape[0], :] = label

        data["label"] = label_pad
        data["ego_waypoint"] = ego_waypoint

        # other measurement
        # do you use the last frame that already happend or use the next frame?
        data["steer"] = measurements[self.seq_len - 1]["steer"]
        data["throttle"] = measurements[self.seq_len - 1]["throttle"]
        data["brake"] = measurements[self.seq_len - 1]["brake"]
        data["light"] = measurements[self.seq_len - 1]["light_hazard"]
        data["speed"] = measurements[self.seq_len - 1]["speed"]
        data["theta"] = measurements[self.seq_len - 1]["theta"]
        data["x_command"] = measurements[self.seq_len - 1]["x_command"]
        data["y_command"] = measurements[self.seq_len - 1]["y_command"]

        # target points
        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        ego_theta = (
            measurements[self.seq_len - 1]["theta"] + rad
        )  # + rad for augmentation
        ego_x = measurements[self.seq_len - 1]["x"]
        ego_y = measurements[self.seq_len - 1]["y"]
        x_command = measurements[self.seq_len - 1]["x_command"]
        y_command = measurements[self.seq_len - 1]["y_command"]

        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
            ]
        )
        local_command_point = np.array([x_command - ego_x, y_command - ego_y])
        local_command_point = R.T.dot(local_command_point)

        data["target_point"] = local_command_point

        data["target_point_image"] = draw_target_point(local_command_point)
        return data


def scale_image(image, scale):
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    return im_resized


def scale_image_cv2(image, scale):
    (width, height) = (int(image.shape[1] // scale), int(image.shape[0] // scale))
    im_resized = cv2.resize(image, (width, height))
    return im_resized


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
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


def crop_image_cv2(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


def get_waypoints(labels, len_labels):
    assert len(labels) == len_labels
    num = len_labels
    waypoints = {}

    for result in labels[0]:
        car_id = result["id"]
        waypoints[car_id] = [[result["ego_matrix"], True]]
        for i in range(1, num):
            for to_match in labels[i]:
                if to_match["id"] == car_id:
                    waypoints[car_id].append([to_match["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints


def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    T = get_vehicle_to_virtual_lidar_transform()

    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix

    return waypoints


def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T


def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())


def parse_labels(labels, rad=0):
    bboxes = {}
    for result in labels:
        num_points = result["num_points"]
        distance = result["distance"]

        x = result["position"][0]
        y = result["position"][1]

        bbox = (
            result["extent"]
            + result["position"]
            + [result["yaw"], result["speed"], result["brake"]]
        )
        bbox = get_bbox_label(bbox, rad)

        # Filter bb that are outside of the LiDAR after the random augmentation. The bounding box is now in image space
        if (
            num_points <= 1
            or bbox[0] <= 0.0
            or bbox[0] >= 255.0
            or bbox[1] <= 0.0
            or bbox[1] >= 255.0
        ):
            continue

        bboxes[result["id"]] = bbox
    return bboxes


def draw_target_point(target_point, color=(255, 255, 255)):
    image = np.zeros((256, 256), dtype=np.uint8)
    target_point = target_point.copy()

    # convert to lidar coordinate
    target_point[1] += 1.3
    point = target_point * 8.0
    point[1] *= -1
    point[1] = 256 - point[1]
    point[0] += 128
    point = point.astype(np.int32)
    point = np.clip(point, 0, 256)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    image = image.reshape(1, 256, 256)
    return image.astype(np.float) / 255.0


def get_bbox_label(bbox, rad=0):
    # dx, dy, dz, x, y, z, yaw
    # ignore z
    dz, dx, dy, x, y, z, yaw, speed, brake = bbox

    pixels_per_meter = 8

    # augmentation
    degree_matrix = np.array(
        [[np.cos(rad), np.sin(rad), 0], [-np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
    )
    T = get_lidar_to_bevimage_transform() @ degree_matrix
    position = np.array([x, y, 1.0]).reshape([3, 1])
    position = T @ position

    position = np.clip(position, 0.0, 255.0)
    x, y = position[:2, 0]
    # center_x, center_y, w, h, yaw
    bbox = np.array([x, y, dy * pixels_per_meter, dx * pixels_per_meter, 0, 0, 0])
    bbox[4] = yaw + rad
    bbox[5] = speed
    bbox[6] = brake
    return bbox


def get_lidar_to_bevimage_transform():
    # rot
    T = np.array([[0, -1, 16], [-1, 0, 32], [0, 0, 1]], dtype=np.float32)
    # scale
    T[:2, :] *= 8

    return T
