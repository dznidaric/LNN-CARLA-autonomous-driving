import glob
import math
import os
import random
import sys
from queue import Empty, Queue

import cv2
import matplotlib.pyplot as plt
import numpy as np
from kerasncp.tf import LTCCell
from kerasncp.wirings import NCP
from ncps.tf import LTC
from PIL import Image
from tensorflow.keras.models import load_model

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

PREFERRED_SPEED = 60
SPEED_THRESHOLD = 5

# max angle when tarining images were produced
YAW_ADJ_DEGREES = 35
MAX_STEER_ANGLE = 35

#mount point of camera on the car
CAMERA_POS_Z = 1.6
CAMERA_POS_X = 0.9

HEIGHT = 360
WIDTH = 640

HEIGHT_REQUIRED_PORTION = 0.4 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.5

# image crop - same as in model input
height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
width_to = width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)

#adding params to display text to image
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (30, 30)
org2 = (30, 50)
fontScale = 0.5
# white color
color = (255, 255, 255)
# Line thickness
thickness = 1


model = load_model("Lane_model/models/ltc_model-0.017563.h5", custom_objects={"LTCCell": LTCCell},compile=False)
model.compile()


def sensor_callback(sensor_data, sensor_queue, sensor_name):

    image = np.reshape(np.copy(sensor_data.raw_data),(sensor_data.height,sensor_data.width,4))
    sensor_queue.put((image, sensor_name))


def camera_rgb_install():
    camera_transform = carla.Transform(
            carla.Location(x=CAMERA_POS_X, z=CAMERA_POS_Z)
        )
    camera_blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_blueprint.set_attribute("image_size_x", '640')
    camera_blueprint.set_attribute("image_size_y", '360')
    camera = world.spawn_actor(
        camera_blueprint, camera_transform, attach_to=ego_vehicle
    )
    
    camera.listen(lambda image: sensor_callback(image, sensor_queue, "camera"))
    actor_list.append(camera)
    sensor_list.append(camera)

def maintain_speed(s):
    ''' 
    this is a very simple function
    too maintan desired speed
    s arg is actual current speed
    '''
    if s >= PREFERRED_SPEED:
        return 0
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.8
    else:
        return 0.3


def predict_angle(im):
    # tweaks for prediction
    img = np.float32(im)
    img_gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    img_gry = cv2.resize(img_gry, (WIDTH,HEIGHT))
    # this version adds taking lower side of the image
    img_gry = img_gry[height_from:,width_from:width_to]
    img_gry = img_gry.astype(np.uint8)
    canny = cv2.Canny(img_gry,50,150)

    #cv2.imshow('processed image',canny)
    canny = canny /255
    input_for_model = canny[ :, :, None] 
    input_for_model = np.expand_dims(input_for_model, axis=0)
    #print('input shape: ',input_for_model.shape)
    angle = model(input_for_model,training=False)
    
    return  angle.numpy()[0][0] * YAW_ADJ_DEGREES / MAX_STEER_ANGLE


actor_list = []

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
client.load_world("Town05")


# Once we have a client we can retrieve the world that is currently running.
world = client.get_world()

#world.set_weather(carla.WeatherParameters.WetNoon)

try:
    # We need to save the settings to be able to recover them at the end
    # of the script to leave the server in the same state that we found it.
    original_settings = world.get_settings()
    settings = world.get_settings()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    # We set CARLA syncronous mode
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    world.apply_settings(settings)

    sensor_queue = Queue()

    spectator = world.get_spectator()

    blueprint_library = world.get_blueprint_library()


    vehicle_blueprints = blueprint_library.filter("*vehicle*")

    town_map = world.get_map()
    good_roads = [37]
    spawn_points = town_map.get_spawn_points()
    good_spawn_points = []
    for point in spawn_points:
        this_waypoint = town_map.get_waypoint(point.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
        if this_waypoint.road_id in good_roads:
            good_spawn_points.append(point)

    
    start_point = random.choice(good_spawn_points)
    ego_vehicle = world.spawn_actor(
        blueprint_library.filter("etron")[0], start_point
    )
    actor_list.append(ego_vehicle)


    """ for i in range(0, 50):
        npc_vehicle = world.try_spawn_actor(
            random.choice(vehicle_blueprints), random.choice(spawn_points)
        )
        if npc_vehicle is not None:
            npc_vehicle.set_autopilot(True)
            actor_list.append(npc_vehicle) """


    sensor_list = []
    # kamera RGB
    camera_rgb_install()

    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    x_pos = 0
    y_pos = 0
    print(ego_vehicle.get_transform().rotation.yaw)
    if ego_vehicle.get_transform().rotation.yaw == 0.0:
        x_pos = -5
    else:
        y_pos = -5

        
    camera_transform = carla.Transform(
        carla.Location(x=x_pos,y=y_pos, z=4),
        carla.Rotation(pitch=-20, yaw=ego_vehicle.get_transform().rotation.yaw),
    )
    camera_bp.set_attribute("image_size_x", '640')
    camera_bp.set_attribute("image_size_y", '360')
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    camera.listen(lambda image: sensor_callback(image, sensor_queue, "3rd person camera"))
    sensor_list.append(camera)

    distance_traveled = 0.0
    lane_crossings = 0
    collisions = 0
    previous_lane_id = None

    time_steps = []
    distances = []
    crossings = []
    collision_count = []

    # Lane invasion sensor
    bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego_vehicle)
    sensor.listen(lambda event: on_invasion(event))
    
    def on_invasion(event):
        global lane_crossings
        # Handle lane invasion event here
        lane_crossings += 1
        print("Lane invasion detected:", event)

    # Collision sensor
    coll_bp = world.get_blueprint_library().find('sensor.other.collision')
    coll_sensor = world.spawn_actor(coll_bp, carla.Transform(), attach_to=ego_vehicle)
    coll_sensor.listen(lambda event: on_collision(event))

    def on_collision(event):
        global collisions
        # Handle collision event here
        collisions += 1
        cv2.imwrite('collision_%{collisions}.jpg',np.array(event.frame))
        print("Collision detected:", event)



    while True:
    # Carla Tick
        world.tick()
        if cv2.waitKey(1) == ord('q'):
            quit = True
            break

        try:
            s_frame = sensor_queue.get(True, 1.0)
            image = s_frame[0]

            predicted_angle = predict_angle(image)
            image = cv2.putText(image, 'Predicted angle in lane: '+str(int(predicted_angle * 90)), org, font, fontScale, color, thickness, cv2.LINE_AA)

            v = ego_vehicle.get_velocity()
            a = ego_vehicle.get_acceleration()

            speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),0)
            image = cv2.putText(image, 'Speed: '+str(int(speed)), org2, font, fontScale, color, thickness, cv2.LINE_AA)

            acceleration = round(math.sqrt(a.x**2 + a.y**2 + a.z**2),1)
            estimated_throttle = maintain_speed(speed)

            location = ego_vehicle.get_location()
            previous_location = location
            if previous_location:
                distance_traveled += location.distance(previous_location)


            time_steps.append(world.get_snapshot().timestamp.elapsed_seconds)
            distances.append(distance_traveled)
            crossings.append(lane_crossings)
            collision_count.append(collisions)
            
            ego_vehicle.apply_control(
                carla.VehicleControl(steer=-predicted_angle, throttle=estimated_throttle)
            )
            #cv2.imshow('RGB Camera',image)

            camera_3rd = sensor_queue.get(True, 1.0)

            if(camera_3rd[0] is not None):
                cv2.imshow('3rd person',np.array(camera_3rd[0]))
            

        except Empty:
            print("Some of the sensor information is missed")

finally:
    cv2.destroyAllWindows()
    world.apply_settings(original_settings)
    for actor in actor_list:
        actor.destroy()
    for sensor in world.get_actors().filter('*sensor*'):
        sensor.destroy()


    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 12))

    # Plot lane crossings
    ax1.plot(time_steps, crossings, label='Prijelazi preko crte', color='orange')
    ax1.set_xlabel('Vrijeme (s)')
    ax1.set_ylabel('Broj prijelaza preko crte')
    ax1.legend(loc='upper left')

    # Add number of collisions to the plot
    ax1.twinx()  # Create a twin y-axis
    ax1.plot(time_steps, collision_count, label='Sudari', color='red')
    ax1.set_ylabel('Broj sudara')
    ax1.legend(loc='upper right')

    ax1.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()
    print("done.")
