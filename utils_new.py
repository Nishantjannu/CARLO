import numpy as np
import os
from gym_carlo.envs.geometry import Point



def determine_segment(map_height, map_width, lane_width, velocity, delta_t, ind):
    """
    Returns segment 1,2 or 3 depending on the index
    """
    unit_step = velocity * delta_t
    # Get segment lengths - a bit inefficient to calculate this all the time over and over
    segment_1_length = map_height/2 - lane_width/2
    n_seg1 = np.arange(0, segment_1_length, unit_step).shape[0]
    segment_2_length = (2*np.pi*lane_width/2) / 4
    n_seg2 = int(segment_2_length / unit_step) + 1
    segment_3_length = map_width/2 - lane_width/2
    n_seg3 = np.arange(0, segment_3_length, unit_step).shape[0]

    # TODO think about edge cases
    print("ind", ind, "n_seg1:", n_seg1, "n_seg1+n_seg2:", n_seg1+n_seg2)
    if ind < n_seg1:
        return 1
    elif n_seg1 <= ind < n_seg1+n_seg2:
        return 2
    else:
        return 3

# NEW
def optimal_trajectory(map_height, map_width, lane_width, velocity, delta_t):
    """
    Generates the optimal (x, y, angle) poses across a trajectory
    as a n x 3 np array

    # TODO:
    - handle edge cases between the segments
    - look at if n_poses_i isnt an integer for any segment
    """
    unit_step = velocity * delta_t
    # Get segment lengths
    segment_1_length = map_height/2 - lane_width/2
    segment_2_length = (2*np.pi*lane_width/2) / 4
    segment_3_length = map_width/2 - lane_width/2

    # First segment, straight up
    # problem with slight offset that builds here!! when using linspace
    # n_poses_1 = int((segment_1_length-0) / unit_step)
    y_poses_1 = np.arange(0, segment_1_length, unit_step).reshape((-1, 1))  # np.linspace(0, segment_1_length, n_poses_1).reshape((n_poses_1, 1))
    x_poses_1 = np.ones((y_poses_1.shape[0], 1))*map_width/2
    angles_1 = np.ones((y_poses_1.shape[0], 1))*(np.pi/2)

    # Second segment, circle
    circle_center = np.array([map_width/2 + lane_width/2, map_height/2 - lane_width/2])  # from geometry of our right turn
    r = lane_width/2
    n_poses_2 = int(segment_2_length / unit_step) + 1
    x_poses_2 = []
    y_poses_2 = []
    angles_2 = []
    for i in range(n_poses_2):
        pos_ang = np.pi - ((np.pi/2) * i / n_poses_2)
        steer_ang = (np.pi/2) - ((np.pi/2) * i / n_poses_2)
        x_poses_2.append( circle_center[0] + r*np.cos(pos_ang) )
        y_poses_2.append( circle_center[1] + r*np.sin(pos_ang) )
        angles_2.append(steer_ang)
    x_poses_2 = np.array(x_poses_2).reshape((n_poses_2, 1))
    y_poses_2 = np.array(y_poses_2).reshape((n_poses_2, 1))
    angles_2 = np.array(angles_2).reshape((n_poses_2, 1))

    # Third segment, straight right
    # n_poses_3 = np.int((segment_3_length-0) / unit_step)
    x_poses_3 = np.arange(map_width/2 + lane_width/2, map_width/2 + lane_width/2 + segment_3_length, unit_step).reshape(-1, 1)  # np.linspace(map_width/2 + lane_width/2, map_width/2 + lane_width/2 + segment_3_length, n_poses_3).reshape((n_poses_3, 1))
    y_poses_3 = np.ones((x_poses_3.shape[0], 1))*map_height/2
    angles_3 = np.zeros((x_poses_3.shape[0], 1))

    # Merge these and return
    x_poses = np.vstack((x_poses_1, x_poses_2, x_poses_3))
    y_poses = np.vstack((y_poses_1, y_poses_2, y_poses_3))
    angles = np.vstack((angles_1, angles_2, angles_3))
    out = np.hstack((x_poses, y_poses, angles))
    return out

def calc_offset(poses, curr_pose, map_height, map_width, lane_width, velocity, delta_t, ind):
    """
    Returns the [delta_psi, e]
        based on the given pose and the optimal trajectory and index
    Note!
        U_y, r are given by the true dynamics instead!!

    curr_pose = [x, y, heading]  # should be given from the scenario

    U_y is the lateral speed
    r is the yaw rate
    delta_psi is the heading angle error
    e is the lateral error from the path
    """
    x, y, heading = curr_pose
    opt_pose = poses[ind, :]
    opt_x, opt_y, opt_heading = opt_pose

    print("curr_pose:", curr_pose, "opt_pose:", opt_pose)

    # delta_psi
    delta_psi = heading - opt_heading

    # e
    segment = determine_segment(map_height, map_width, lane_width, velocity, delta_t, ind)
    print("segment:", segment)
    assert segment in [1, 2, 3], "Unknown segment!"
    if segment == 1:
        e = x - opt_x
    elif segment == 2:
        circle_center = np.array([map_width / 2 + lane_width / 2, map_height/2 - lane_width/2])
        r = np.sqrt((x-circle_center[0])**2 + (y-circle_center[1])**2)
        r_opt = np.sqrt((opt_x-circle_center[0])**2 + (opt_y-circle_center[1])**2)
        e = r - r_opt
    elif segment == 3:
        e = y - opt_y

    print("e:", e, "delta_psi", delta_psi)
    return e, delta_psi


# OLD
scenario_names = ['intersection', 'circularroad', 'lanechange']
obs_sizes = {'intersection': 5, 'circularroad': 4, 'lanechange': 3}
goals = {'intersection': ['left','straight','right'], 'circularroad': ['inner','outer'], 'lanechange': ['left','right']}
steering_lims = {'intersection': [-0.5,0.5], 'circularroad': [-0.15,0.15], 'lanechange': [-0.15, 0.15]}



def optimal_act_circularroad(env, d):
    if env.ego.speed > 10:
        throttle = 0.06 + np.random.randn()*0.02
    else:
        throttle = 0.6 + np.random.randn()*0.1

    # setting the steering is not fun. Let's practice some trigonometry
    r1 = 30.0 # inner building radius (not used rn)
    r2 = 39.2 # inner ring radius
    R = 32.3 # desired radius
    if d==1: R += 4.9
    Rp = np.sqrt(r2**2 - R**2) # distance between current "target" point and the current desired point
    theta = np.arctan2(env.ego.y - 60, env.ego.x - 60)
    target = Point(60 + R*np.cos(theta) + Rp*np.cos(3*np.pi/2-theta), 60 + R*np.sin(theta) - Rp*np.sin(3*np.pi/2-theta)) # this is pure magic (or I need to draw it to explain)
    desired_heading = np.arctan2(target.y - env.ego.y, target.x - env.ego.x) % (2*np.pi)
    h = np.array([env.ego.heading, env.ego.heading - 2*np.pi])
    hi = np.argmin(np.abs(desired_heading - h))

    if desired_heading >= h[hi]:
        steering = 0.15 + np.random.randn()*0.05
    else:
        steering = -0.15 + np.random.randn()*0.05
    return np.array([steering, throttle]).reshape(1,-1)


def optimal_act_lanechange(env, d):
    if env.ego.speed > 10:
        throttle = 0.06 + np.random.randn()*0.02
    else:
        throttle = 0.8 + np.random.randn()*0.1

    if d==0:
        target = Point(37.55, env.ego.y + env.ego.speed*3)
    elif d==1:
        target = Point(42.45, env.ego.y + env.ego.speed*3)
    desired_heading = np.arctan2(target.y - env.ego.y, target.x - env.ego.x) % (2*np.pi)
    h = np.array([env.ego.heading, env.ego.heading - 2*np.pi])
    hi = np.argmin(np.abs(desired_heading - h))

    if desired_heading >= h[hi]:
        steering = 0.15 + np.random.randn()*0.05
    else:
        steering = -0.15 + np.random.randn()*0.05

    return np.array([steering, throttle]).reshape(1,-1)
