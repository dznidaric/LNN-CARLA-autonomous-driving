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
from train_LNN_model import LTC_CNN3D_model

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
CAMERA_LEFT = -60.0
CAMERA_RIGHT = 60.0

# IMAGE
IMAGE_WIDTH = 320
IMAGE_RESOLUTION = (160, 704)

COMBINED_IMAGE_SHAPE = (160, 704, 3)

MODEL_INPUT_SIZE = (10,) + COMBINED_IMAGE_SHAPE

MODEL_OUTPUT_SIZE = (5,)

model = LTC_CNN3D_model()
#model.load_weights("models/LTC_CNN3D_model-0.9783.hdf5")
model = load_model("models/LTC_CNN2D_model-0.8464.h5", custom_objects={"LTCCell": LTCCell})

def process_img(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3]
    """ cv2.imshow("", i3)
    cv2.waitKey(1) """
    i3 = cv2.cvtColor(i3, cv2.COLOR_BGR2RGB)
    i3 = Image.fromarray(i3)
    (width, height) = (i3.width // scale, i3.height // scale)
    if crop_x is None:
        crop_x = width
    if crop_y is None:
        crop_y = height

    i4 = np.asarray(i3)
    cropped_image = i4[start_y : start_y + crop_y, start_x : start_x + crop_x]
    return cropped_image / 255.0


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    processed_img = process_img(
        sensor_data,
        1,
        IMAGE_WIDTH,
        IMAGE_WIDTH,
        IMAGE_RESOLUTION[0],
        IMAGE_RESOLUTION[0],
    )
    sensor_queue.put((processed_img, sensor_name))


sensor_list = []


def camera_rgb_install():
    camera_positions = [
        carla.Transform(
            carla.Location(x=CAMERA_POSITION[0], z=CAMERA_POSITION[2]),
            carla.Rotation(yaw=CAMERA_LEFT),
        ),  # Left
        carla.Transform(
            carla.Location(x=CAMERA_POSITION[0], z=CAMERA_POSITION[2])
        ),  # Center
        carla.Transform(
            carla.Location(x=CAMERA_POSITION[0], z=CAMERA_POSITION[2]),
            carla.Rotation(yaw=CAMERA_RIGHT),
        ),
    ]  # Right

    cameras = []
    for camera_transform in camera_positions:
        camera_blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_blueprint.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_blueprint.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_blueprint.set_attribute("fov", str(CAMERA_FOV))
        # camera_blueprint.set_attribute("sensor_tick", "0.1")
        camera = world.spawn_actor(
            camera_blueprint, camera_transform, attach_to=ego_vehicle
        )
        camera.listen(
            lambda image: sensor_callback(
                image,
                sensor_queue,
                "camera%d" % camera_positions.index(camera_transform),
            )
        )
        actor_list.append(camera)
        cameras.append(camera)
        sensor_list.append(camera)
    return cameras


actor_list = []

client = carla.Client("localhost", 2000)
client.set_timeout(3.0)

# Once we have a client we can retrieve the world that is currently running.

#world = client.load_world("Town04")
world = client.get_world()

try:
    # We need to save the settings to be able to recover them at the end
    # of the script to leave the server in the same state that we found it.
    original_settings = world.get_settings()
    settings = world.get_settings()

    # We set CARLA syncronous mode
    settings.fixed_delta_seconds = 0.05  # Fixed time-step == 10 FPS (1 / 0,1)
    settings.synchronous_mode = True
    world.apply_settings(settings)

    sensor_queue = Queue()

    # Retrieve the spectator object
    spectator = world.get_spectator()

    blueprint_library = world.get_blueprint_library()

    # Set the spectator with an empty transform
    # spectator.set_transform(carla.Transform())

    # This will set the spectator at the origin of the map, with 0 degrees
    # pitch, yaw and roll - a good way to orient yourself in the map

    vehicle_blueprints = blueprint_library.filter("*vehicle*")

    spawn_points = world.get_map().get_spawn_points()

    
    spawn_point_selected = random.choice(spawn_points)
    ego_vehicle = world.spawn_actor(
        blueprint_library.filter("etron")[0], spawn_point_selected
    )
    actor_list.append(ego_vehicle)

    print("spawn_location: ", spawn_point_selected)

    for i in range(0, 50):
        npc_vehicle = world.try_spawn_actor(
            random.choice(vehicle_blueprints), random.choice(spawn_points)
        )
        if npc_vehicle is not None:
            npc_vehicle.set_autopilot(True)
            actor_list.append(npc_vehicle)

    """ postavljanje senzora na auto """
    # kamera RGB
    camera_list = camera_rgb_install()

    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_transform = carla.Transform(carla.Location(x=0, z=25), carla.Rotation(pitch=-90, yaw = spawn_point_selected.rotation.yaw*90))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    seconds = 720
    image_queue = np.empty((10, 160, 704, 3), dtype=np.float32)
    curr_wp = 5
    brojac = 0
    for sec in range(seconds):
        # Synchronize the CARLA world
        world.tick()
        w_frame = world.get_snapshot().frame

        combined_image = np.empty(COMBINED_IMAGE_SHAPE, dtype=np.float32)
        camera_frames = []

        try:
            for i in range(len(sensor_list)):
                s_frame = sensor_queue.get(True, 1.0)
                camera_frames.append(s_frame[0])
            combined_image = np.concatenate(np.array(camera_frames), axis=1)
            combined_image = cv2.resize(
                combined_image, dsize=(704, 160), interpolation=cv2.INTER_LINEAR
            )
            # cv2.imwrite("images/combined_image.png", combined_image)

            image_queue[brojac] = combined_image
            brojac += 1

            if brojac == 10:
                #cv2.imwrite("images/image_queue.png", image_queue[0] * 255.0)
                result = model.predict(np.expand_dims(image_queue, axis=0))
                commands = result[0][9]
                steer = round(commands[0].item(),1)
                throttle = round(commands[1].item(),1)
                brake = round(commands[2].item(),1)

                if steer == -0.0:
                    steer = 0.0
                print("Steer: ", steer)
                print("throttle: ", throttle)
                print("brake: ", brake)
                if brake < 0.5:
                    brake = 0.0
                ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                spectator.set_transform(camera.get_transform())
                #spectator.set_transform(carla.Transform(carla.Location(x=spawn_point_selected.location.x, y=spawn_point_selected.location.y , z=15), carla.Rotation(pitch=-60, yaw = spawn_point_selected.rotation.yaw+90)))
                brojac = 0
                image_queue = np.empty((10, 160, 704, 3), dtype=np.float32)

        except Empty:
            print("    Some of the sensor information is missed")

finally:
    world.apply_settings(original_settings)
    for actor in actor_list:
        actor.destroy()
    print("done.")
