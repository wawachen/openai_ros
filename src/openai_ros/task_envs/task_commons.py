#!/usr/bin/env python
import rosparam
import rospkg
import os
from itertools import tee
import numpy as np

def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)

##generate different trajectories
def figure8_trajectory(start_x, start_z, amp, num_points_per_rot=100):
    length = np.pi*2/np.pi
    theta_inc = length / num_points_per_rot

    waypoints = []

    for i in range(num_points_per_rot):
        theta_x = i * theta_inc

        nextpt = [start_x-length/2, start_z]
        nextpt[1] += amp * np.sin(theta_x*np.pi)
        nextpt[0] += theta_x 

        waypoints.append(np.array(nextpt))
    
    for i in range(num_points_per_rot):
        theta_x = i * theta_inc

        nextpt = [start_x+length/2, start_z]
        nextpt[1] += amp * np.sin(theta_x*np.pi)
        nextpt[0] -= theta_x 

        waypoints.append(np.array(nextpt))

    return waypoints

##generate different trajectories
def figure8_trajectory_3d(start_x, start_z, amp, num_points_per_rot=100):
    length = np.pi*2/np.pi
    theta_inc = length / num_points_per_rot

    waypoints = []

    for i in range(num_points_per_rot):
        theta_x = i * theta_inc

        nextpt = [start_x-length/2, 0.0, start_z]
        nextpt[2] += amp * np.sin(theta_x*np.pi)
        nextpt[0] += theta_x 

        waypoints.append(np.array(nextpt))
    
    for i in range(num_points_per_rot):
        theta_x = i * theta_inc

        nextpt = [start_x+length/2, 0.0, start_z]
        nextpt[2] += amp * np.sin(theta_x*np.pi)
        nextpt[0] -= theta_x 

        waypoints.append(np.array(nextpt))

    return waypoints

##generate different trajectories
def figure8_trajectory_3d_xy(start_x, start_z, amp, num_points_per_rot=100):
    length = np.pi*2/np.pi
    theta_inc = length / num_points_per_rot

    waypoints = []

    for i in range(num_points_per_rot):
        theta_x = i * theta_inc

        nextpt = [start_x-length/2, start_z, 1.0]
        nextpt[1] += amp * np.sin(theta_x*np.pi)
        nextpt[0] += theta_x 

        waypoints.append(np.array(nextpt))
    
    for i in range(num_points_per_rot):
        theta_x = i * theta_inc

        nextpt = [start_x+length/2, start_z, 1.0]
        nextpt[1] += amp * np.sin(theta_x*np.pi)
        nextpt[0] -= theta_x 

        waypoints.append(np.array(nextpt))

    return waypoints

######## trajectory ########
class Trajectory:
    # metric represents evaluation
    def __init__(self, iterator, metric, is_waiting_func, loop=False):
        _, self.original, self.iterator = tee(iterator, 3)
        self.metric = metric
        self.is_waiting = is_waiting_func
        self.curr_goal = None
        self.loop = loop
        self.completed = False  # if we have moved through the whole trajectory
        self.i = 0

    def next(self, state):
        # ensures that goal is set after at least one call to next (used by pickup code)
        # self.metric(state, state, self.i)

        # metric first
        if self.curr_goal is None or self.metric(state, self.curr_goal, self.i):
            # in this case, we have reached the current goal
            try:
                self.curr_goal = next(self.iterator)
                self.i += 1  # represents the number of distinct targets we have seen
            except StopIteration as e:
                if self.loop:
                    self.reset()
                    try:
                        self.curr_goal = next(self.iterator)
                    except StopIteration as e:
                        # in this case we cannot loop around so just give up
                        self.completed = True
                else:
                    self.completed = True

        # goals are always relative to [0,0,0], not true origin
        # states are always relative to true origin
        # we correct for that here
        return self.curr_goal

    # rolls out goals for steps without affecting the original
    def try_next_n(self, num):
        _, self.iterator, try_iter = tee(self.iterator, 3)

        goals = []
        try:
            for j in range(num):
                # stop if we ever reach a waiting point (corresponds to the point before)
                if self.is_waiting(self.i + j):
                    raise StopIteration
                next_goal = next(try_iter)
                goals.append(next_goal)
        except StopIteration:
            # logger.debug("stopped at %d elements" % len(goals))
            if len(goals) == 0 and self.curr_goal is not None:
                goals = [self.curr_goal] * num
            elif len(goals) > 0:
                last_filled = len(goals) - 1
                for i in range(num - len(goals)):
                    goals.append(goals[last_filled])  # fill the remainder
            else:
                raise NotImplementedError("[Traj]: try next n not defined when curr_goal is None")

        return np.stack(goals)

    def get_i(self):
        return self.i

    def is_finished(self):
        return self.completed

    def reset(self, zero_at=None, new_iterator=None):
        if new_iterator is not None:
            self.iterator = new_iterator
        else:
            _, self.original, self.iterator = tee(self.original, 3)

        self.curr_goal = None
        self.completed = False
        self.i = 0

        # logger.debug("RESET AT: " + str(self.origin[:2]))
        
    
class Metrics:
    @staticmethod
    def distance_thresh(thresh, mask=None):

        def func(state, goal, i):
            nonlocal mask
            if mask is None:
                mask = np.ones(state.shape)
            return np.linalg.norm(np.multiply(state, mask) - goal)

        return Metrics.ext_function_thresh(thresh, func)

    @staticmethod
    def individual_distance_thresh(thresh_array, mask=None):

        def func(state, goal, i):
            nonlocal mask
            if mask is None:
                mask = np.ones(state.shape)
            abs_diff = np.abs(np.multiply(state, mask) - goal)
            # logger.debug("[TEMP] offsets / within: " + str(abs_diff) + " / " + str(abs_diff <= thresh_array))
            return np.all(abs_diff <= thresh_array)

        return func

    @staticmethod
    def sequential():
        return lambda state, goal, i: True

    @staticmethod
    def wait_start_sequential(initial_func):
        # runs sequentially after the first
        # return lambda state, goal, i: initial_func(state, goal, i) if i <= 1 else True
        return lambda state, goal, i: initial_func(state, goal, i) if i <0 else True

    @staticmethod
    def ext_function_thresh(thresh, func):

        def metric(state, goal, i):
            assert state.shape == goal.shape
            dist = func(state)
            # termination condition
            return dist < thresh

        return metric