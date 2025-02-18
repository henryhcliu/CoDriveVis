import time
import carla
import math
import numpy as np
from collections import deque

def draw_planned_trj(world, x_trj, car_z, color=(200,0,0)):
    color = carla.Color(r=color[0],g=color[1],b=color[2],a=0)
    length = x_trj.shape[0]
    xx = x_trj[:,0]
    yy = x_trj[:,1]
    for i in range(1, length):
        world.debug.draw_point(carla.Location(x=xx[i], y=yy[i], z=car_z), size=0.1, life_time=5)
        time.sleep(0.01)

def carla_vector_to_rh_vector(position, yaw, velocity=None):
    """
    Convert a carla location to a right-hand system
    position: x, y, z
    yaw: yaw(degree)
    velocity: vx, vy, omega
    """
    x = position[0]
    y = -position[1]
    yaw = -np.radians(yaw)

    if velocity is not None:
        vx = velocity[0]
        vy = -velocity[1]
        omega = -velocity[2]
        
        return [x, y, yaw, vx, vy, omega]
    
    return [x, y, yaw, 0, 0, 0]

def generate_spawn_points_nearby(world, given_transform, max_dis, min_dis, spawn_points, numbers_of_vehicles):
    """
    parameters:
    ego_vehicle :: your target vehicle
    max_dis :: the distance max limitation between ego-vehicle and other free-vehicles
    min_dis :: the distance min limitation
    spawn_points :: the available spawn points in current map
    numbers_of_vehicles :: the number of free-vehicles around ego-vehicle that you need
    """
    np.random.shuffle(spawn_points) # shuffle all the spawn points
    ego_location = given_transform.location
    accessible_points = []
    for spawn_point in spawn_points:
        dis = math.sqrt((ego_location.x-spawn_point.location.x)**2 + (ego_location.y-spawn_point.location.y)**2)
        # generate a waypoint near the spawn point
        waypoint = world.get_map().get_waypoint(spawn_point.location)
        # print("is junction: ", waypoint.is_junction)
        # it also can include z-coordinate,but it is unnecessary
        if dis < max_dis and dis > min_dis and (spawn_point not in accessible_points) and waypoint.is_junction == False:
            accessible_points.append(spawn_point)

    transform_list = [] # keep the spawned vehicle in vehicle_list, because we need to link them with traffic_manager
    if len(accessible_points) < numbers_of_vehicles:
        # if your radius is relatively small,the satisfied points may be insufficient
        numbers_of_vehicles = len(accessible_points)

    for i in range(numbers_of_vehicles): # generate the free vehicle
        point = accessible_points[i]
        transform_list.append(point)

    # print("spawned vehicles number: ", len(transform_list))
    return transform_list

class PIDAccelerationController():
    """
    PIDAccelerationController implements acceleration control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_acc, debug=False):
        """
        Execute one step of acceleration control to reach a given target speed.

            :param target_acceleration: target acceleration in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_acc = self.get_acc()

        if debug:
            print('Current acceleration = {}'.format(current_acc))

        # print('err', current_acc, target_acc, target_acc-current_acc)
        return self._pid_control(target_acc, current_acc)

    def _pid_control(self, target_acc, current_acc):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """
        error = target_acc - current_acc
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

    def get_acc(self):
        # direction flag, 1: forward, -1: backward
        flag = 1

        yaw = np.radians(self._vehicle.get_transform().rotation.yaw)
        ax = self._vehicle.get_acceleration().x
        ay = self._vehicle.get_acceleration().y 
        acc_yaw = math.atan2(ay, ax)
        error = acc_yaw - yaw
        if error > math.pi:
            error -= 2 * math.pi
        elif error < -math.pi:
            error += 2 * math.pi
        error = math.fabs(error)
        if error > math.pi / 2:
            flag = -1

        return flag * np.sqrt(ax**2+ay**2)*0.1