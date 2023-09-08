#from CARLA camera tutorial on YouTube
# this approach make the camera image available with for a simple loop

import math
import os
import random
import time

import cv2
import numpy as np

import carla

# max yaw angle from straight
YAW_ADJ_DEGREES = 35

PREFERRED_SPEED = 10 #optional

#mount point of camera on the car
CAMERA_POS_Z = 1.6
CAMERA_POS_X = 0.9

HEIGHT = 360
WIDTH = 640

HEIGHT_REQUIRED_PORTION = 0.4 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.5

YAW_ADJ_DEGREES = 35

height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
width_to = width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)

# I separately learned road id's covering the ring highway around Town 5
# not this would be different in other Towns/maps
good_roads = [12, 34, 35, 36, 37, 38, 1201, 1236, 2034, 2035, 2343, 2344]


# connect to sim
client = carla.Client('localhost', 2000)
client.set_timeout(15)

# load Town5 map
client.load_world('Town05') 

#transform car through waypoints in a loop while printing the angle onto the image

# sim settings
world = client.get_world()

try:

    traffic_manager = client.get_trafficmanager(8000)
    settings = world.get_settings()
    traffic_manager.set_synchronous_mode(True)
    # option preferred speed
    # traffic_manager.set_desired_speed(vehicle,float(PREFERRED_SPEED))
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # pick a car model - Tesla M3
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('etron')

    town_map = world.get_map()

    # optional - get all intersections to avoid cutting images near them
    junction_list = []
    waypoint_list = town_map.generate_waypoints(2.0)
    for x in waypoint_list:
        if x.get_junction() is not None:
            junction_list.append(x.transform.location)
                    
    #limit spawn points to highways
    spawn_points = town_map.get_spawn_points()
    good_spawn_points = []
    for point in spawn_points:
        this_waypoint = town_map.get_waypoint(point.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
        if this_waypoint.road_id in good_roads:
            good_spawn_points.append(point)

    all_waypoint_pairs = town_map.get_topology()
    # subset of lane start/end's which belong to good roads
    good_lanes = []
    for w in all_waypoint_pairs:
        if w[0].road_id in good_roads:
            good_lanes.append(w)        

    transform = random.choice(good_spawn_points)
    ego_vehicle = world.try_spawn_actor(vehicle_bp[0], transform)

    #setting RGB Camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640') 
    camera_bp.set_attribute('image_size_y', '360')

    camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
    camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=ego_vehicle)

    def camera_callback(image,data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

    image_w = camera_bp.get_attribute('image_size_x').as_int()
    image_h = camera_bp.get_attribute('image_size_y').as_int()

    camera_data = {'image': np.zeros((image_h,image_w,4))}
    camera.listen(lambda image: camera_callback(image,camera_data))

    #main loop 
    quit = False
    images = []
    angles = []
    path = f"64_batched_data_town05"
    num_batches = 0
    for lane in good_lanes:
        #loop within a lane
        if quit:
            break
        for wp in lane[0].next_until_lane_end(20):
            transform = wp.transform
            ego_vehicle.set_transform(wp.transform)
            time.sleep(2) #these delays seem to be necessary for teh car to take the position before a shot is taken
            initial_yaw = wp.transform.rotation.yaw
            # do multiple shots of straight direction
            for i in range(5):
                # Carla Tick
                world.tick()
                
                trans = wp.transform
                angle_adj = random.randrange(-YAW_ADJ_DEGREES, YAW_ADJ_DEGREES, 1)
                trans.rotation.yaw = initial_yaw +angle_adj 
                ego_vehicle.set_transform(trans)
                time.sleep(1)  #these delays seem to be necessary for teh car to take the position before a shot is taken
                if cv2.waitKey(1) == ord('q'):
                    quit = True
                    break
                img = camera_data['image']
                
                actual_angle = ego_vehicle.get_transform().rotation.yaw - initial_yaw
                #fixing values like 361.5 to make them close to zero
                if actual_angle <-180:
                    actual_angle +=360
                elif actual_angle >180:
                    actual_angle -=360
                actual_angle = str(int(actual_angle))
                angle = float(actual_angle)/YAW_ADJ_DEGREES
                angles.append(angle)

                img = np.float32(img)
                img_gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
                image = img_gry[height_from:,width_from:width_to]
                canny = cv2.Canny(np.uint8(image),50,150)
                images.append(canny[:, :, None] / 255)

                time_grab = time.time_ns()
                if(len(images) == 64):
                    try:
                        np.save(f'{path}/images/64_images_{num_batches}', np.array(images))
                        np.save(f'{path}/labels/64_labels_{num_batches}', np.array(angles))
                    except FileNotFoundError:
                        os.makedirs(f"{path}")
                        np.save(f'{path}/images/64_images_{num_batches}', np.array(images))
                        np.save(f'{path}/labels/64_labels_{num_batches}', np.array(angles))
                    num_batches += 1
                    images = []
                    angles = []
                    
                #cv2.imwrite('_out_ang/%06d_%s.png' % (time_grab, actual_angle), img_gry)
                #old way to screen - cv2.imshow('RGB Camera',img)

finally:
    #clean up
    cv2.destroyAllWindows()
    camera.stop()
    for actor in world.get_actors().filter('*ego_vehicle*'):
        actor.destroy()
    for sensor in world.get_actors().filter('*sensor*'):
        sensor.destroy()