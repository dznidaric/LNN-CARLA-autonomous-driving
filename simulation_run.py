import glob
import os
import random
import sys
import time
from queue import Empty, Queue

import cv2
import numpy as np
from kerasncp.tf import LTCCell
from kerasncp.wirings import NCP
from ncps.tf import LTC
from PIL import Image
from tensorflow.keras.models import load_model

from constants_1 import INPUT_SHAPE

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla

CAMERA_POSITION = [1.3, 0.0, 2.3]
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 960
CAMERA_FOV = 120

# IMAGE
IMAGE_WIDTH = 320
IMAGE_RESOLUTION = (160, 704)


model = load_model(
    "models/cnn_ncp_model-0.002525.h5", custom_objects={"LTCCell": LTCCell}
)


def process_img(image):
    i = np.array(image.raw_data)

    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3]
    i4 = cv2.resize(
        i3, (INPUT_SHAPE[1], INPUT_SHAPE[0]), interpolation=cv2.INTER_LINEAR
    )
    return i4


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    processed_img = process_img(sensor_data)
    sensor_queue.put((processed_img, sensor_name))


sensor_list = []


def camera_rgb_install():
    camera_transform = carla.Transform(
        carla.Location(x=CAMERA_POSITION[0], z=CAMERA_POSITION[2])
    )

    camera_blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_blueprint.set_attribute("image_size_x", str(CAMERA_WIDTH))
    camera_blueprint.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    camera_blueprint.set_attribute("fov", str(CAMERA_FOV))
    camera = world.spawn_actor(
        camera_blueprint, camera_transform, attach_to=ego_vehicle
    )
    camera.listen(lambda image: sensor_callback(image, sensor_queue, "camera"))
    actor_list.append(camera)
    sensor_list.append(camera)
    return camera


actor_list = []

client = carla.Client("localhost", 2000)
client.set_timeout(3.0)

# Once we have a client we can retrieve the world that is currently running.

# world = client.load_world("Town01_Opt")
world = client.get_world()

try:
    # We need to save the settings to be able to recover them at the end
    # of the script to leave the server in the same state that we found it.
    original_settings = world.get_settings()
    settings = world.get_settings()

    # We set CARLA syncronous mode
    settings.fixed_delta_seconds = 0.5  # Fixed time-step == 10 FPS (1 / 0,1)
    settings.synchronous_mode = True
    world.apply_settings(settings)

    sensor_queue = Queue()

    # Retrieve the spectator object
    spectator = world.get_spectator()

    blueprint_library = world.get_blueprint_library()

    vehicle_blueprints = blueprint_library.filter("*vehicle*")

    spawn_points = world.get_map().get_spawn_points()

    print(len(spawn_points))
    spawn_point_selected = random.choice(spawn_points)
    ego_vehicle = world.spawn_actor(
        blueprint_library.filter("etron")[0], spawn_points[250]
    )
    actor_list.append(ego_vehicle)

    print("spawn_location: ", spawn_point_selected)

    """ for i in range(0, 50):
        npc_vehicle = world.try_spawn_actor(
            random.choice(vehicle_blueprints), random.choice(spawn_points)
        )
        if npc_vehicle is not None:
            npc_vehicle.set_autopilot(True)
            actor_list.append(npc_vehicle) """

    """ postavljanje senzora na auto """
    # kamera RGB
    camera_rgb_install()

    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_transform = carla.Transform(
        carla.Location(x=0, z=25),
        carla.Rotation(pitch=-90, yaw=spawn_point_selected.rotation.yaw * 90),
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    seconds = 720
    for sec in range(seconds):
        # Synchronize the CARLA world
        world.tick()

        try:
            s_frame = sensor_queue.get(True, 1.0)
            camera_frame = s_frame[0]

            #cv2.imwrite(f"images/image_{int(time.time())}.png", camera_frame)
            result = model.predict(np.expand_dims(camera_frame, axis=0))
            commands = result[0]
            steer = round(commands[0].item(), 1)
            scaled_steer = 2 * steer - 1

            if scaled_steer >= 0.15 or scaled_steer <= -0.15:
                throttle = random.uniform(0.1, 0.2)
            else:
                throttle = random.uniform(0.2, 0.5)
            print("scaled_steer: ", scaled_steer)

            ego_vehicle.apply_control(
                carla.VehicleControl(steer=scaled_steer, throttle=throttle)
            )
            spectator.set_transform(
                carla.Transform(
                    ego_vehicle.get_location() + carla.Location(z=25),
                    carla.Rotation(
                        pitch=-90
                    ),
                )
            )

        except Empty:
            print("    Some of the sensor information is missed")

finally:
    world.apply_settings(original_settings)
    for actor in actor_list:
        actor.destroy()
    print("done.")
