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
        self.scale = np.array(config.scale)

        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.inv_augment_prob = np.array(config.inv_augment_prob)

        self.image_shape = (160, 704, 3)
        self.measurements_shape = (4,)

        self.images = []
        self.labels = []
        self.measurements = []
        self.num_samples = []

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
                num_seq = len(os.listdir(route_dir / "rgb"))

                self.num_samples.append(num_seq - 4)
                # ignore the first two and last two frame
                for seq in range(2, num_seq - 2):
                    self.images.append(route_dir / "rgb" / ("%04d.png" % seq))
                    self.measurements.append(
                        route_dir / "measurements" / ("%04d.json" % seq)
                    )

    def data_generator(self):
        images = self.images
        labels = self.labels
        measurements = self.measurements

        len_parameters = len(self.images)

        # load measurements
        loaded_labels = []

        # Since we also load labels for future timesteps, we load and store them separately
        for i in range(len_parameters):
            """for i in range(self.seq_len + self.pred_len):
                with open(str(labels[i]), "r", encoding="utf-8") as f2:
                    loaded_labels.append(ujson.load(f2))
            labels = loaded_labels"""

            with open(str(measurements[i]), "r", encoding="utf-8") as f1:
                measurement = ujson.load(f1)

            image = cv2.imread(str(images[i]), cv2.IMREAD_COLOR)
            if image is None:
                print("Error loading file: ", str(images[i], encoding="utf-8"))

            """ 
                IMAGE
            """
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.array(image, dtype=np.float32)

            # augmentation
            crop_shift = 0
            degree = 0
            rad = np.deg2rad(degree)
            do_augment = (
                self.augment and random.random() <= self.inv_augment_prob
            )  # if random number is less than or equal to inv_augment_prob (0.15)
            if do_augment:
                degree = (random.random() * 2.0 - 1.0) * self.aug_max_rotation
                rad = np.deg2rad(degree)
                crop_shift = degree / 60 * self.img_width / self.scale  # we scale first

            image = crop_image_cv2(
                image, crop=self.img_resolution, crop_shift=crop_shift
            )

            steer = measurement["steer"]
            throttle = measurement["throttle"]
            brake = measurement["brake"]
            light = measurement["light_hazard"]
            speed = measurement["speed"]
            theta = measurement["theta"]
            x_command = measurement["x_command"]
            y_command = measurement["y_command"]

            self.image_shape = image.shape
            # self.measurements_shape = np.array([steer, throttle, brake, light, speed, theta, x_command, y_command]).shape
            self.measurements_shape = np.array([steer, throttle, brake, light]).shape

            yield {
                "rgb": image,
                "measurements": np.array(
                    [steer, throttle, brake, light], dtype=np.float32
                ),
            }

            """ 
                LABELS
            """
            """ # ego car is always the first one in label file
            ego_id = labels[self.seq_len - 1][0]["id"]  #432

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
            data["ego_waypoint"] = ego_waypoint """

            """ 
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

            data["target_point_image"] = draw_target_point(local_command_point) """

    def _inputs(self):
        # Return the structure of input elements
        return ()  # No inputs in this case

    @property
    def element_spec(self):
        # Return the structure of elements in the dataset
        return {
            "rgb": tf.TensorSpec(shape=self.image_shape, dtype=tf.float32),
            "measurements": tf.TensorSpec(shape=(4,), dtype=tf.float32)
        }

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
