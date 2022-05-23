"""

"""

import numpy as np


class Nominal_Trajectory_Handler:
    """
    Tracks the optimal trajectory. Uses a static varibale current_index to do so

    TODO: maybe keep the optimal trajectory as an internal variable
    """
    # Static variable for tracking current_index
    current_index = 0


    def __init__(self, map_height, map_width, lane_width, velocity, delta_t):
        self.map_height = map_height
        self.map_width = map_width
        self.lane_width = lane_width
        self.unit_step =  velocity * delta_t

        # Calculate the different segments of the road
        unit_step = velocity * delta_t
        self.segment_1_length = map_height/2 - lane_width/2
        self.n_seg1 = np.arange(0, self.segment_1_length, self.unit_step).shape[0]
        self.segment_2_length = (2*np.pi*lane_width/2) / 4
        self.n_seg2 = int(self.segment_2_length / self.unit_step) + 1
        self.segment_3_length = map_width/2 - lane_width/2
        self.n_seg3 = np.arange(0, self.segment_3_length, self.unit_step).shape[0]

    def increment_current_index(self):
        """
        Should only be called once per iteration loop!!
        Increments the static index of the trajectory_handler
        """
        Nominal_Trajectory_Handler.current_index += 1

    def reset_index(self):
        Nominal_Trajectory_Handler.current_index = 0

    def get_segment(self):
        """
        Returns segment 1,2 or 3 depending on the static index
        """
        # TODO think about edge cases(?)
        if Nominal_Trajectory_Handler.current_index < self.n_seg1:
            return 1
        elif n_seg1 <= Nominal_Trajectory_Handler.current_index < self.n_seg1+self.n_seg2:
            return 2
        else:
            return 3

    def get_kappa(self):
        seg = self.get_segment()
        if seg == 1 or seg == 3:
            return 0
        else:
            return 1/(self.lane_width/2)  # 1/r


    def get_current_optimal_pose(self, opt_traj):
        return opt_traj[Nominal_Trajectory_Handler.current_index, :]


    def get_x_y_from_state(self, state, opt_traj):  # state should be true state? to find x. Could also do that calc in true dynamics
        """
        # State #
        x = [U_y, r, delta_psi, e]
        U_y is the lateral speed
        r is the yaw rate
        delta_psi is the heading angle error
        e is the lateral error from the path
        """
        opt_x, opt_y, opt_heading = get_current_optimal_pose(opt_traj)
        y = opt_y + state[3]  # need to adapt for the middle segment


    def get_optimal_trajectory(self):
        """
        Generates the optimal (x, y, angle) poses across a trajectory
        as a n x 3 np array

        # TODO:
        - handle edge cases between the segments
        - look at if n_poses_i isnt an integer for any segment
        """
        # First segment, straight
        y_poses_1 = np.arange(0, self.segment_1_length, self.unit_step).reshape((-1, 1))
        x_poses_1 = np.ones((y_poses_1.shape[0], 1))*self.map_width/2
        angles_1 = np.ones((y_poses_1.shape[0], 1))*(np.pi/2)

        # Second segment, circle
        circle_center = np.array([self.map_width/2 + self.lane_width/2, self.map_height/2 - self.lane_width/2])  # from geometry of our right turn
        r = self.lane_width/2
        x_poses_2 = []
        y_poses_2 = []
        angles_2 = []
        for i in range(self.n_seg2):
            pos_ang = np.pi - ((np.pi/2) * i /self.n_seg2)
            steer_ang = (np.pi/2) - ((np.pi/2) * i / self.n_seg2)
            x_poses_2.append( circle_center[0] + r*np.cos(pos_ang) )
            y_poses_2.append( circle_center[1] + r*np.sin(pos_ang) )
            angles_2.append(steer_ang)
        x_poses_2 = np.array(x_poses_2).reshape((self.n_seg2, 1))
        y_poses_2 = np.array(y_poses_2).reshape((self.n_seg2, 1))
        angles_2 = np.array(angles_2).reshape((self.n_seg2, 1))

        # Third segment, straight right
        x_poses_3 = np.arange(self.map_width/2 + self.lane_width/2, self.map_width/2 + self.lane_width/2 + self.segment_3_length, self.unit_step).reshape(-1, 1)  # np.linspace(map_width/2 + lane_width/2, map_width/2 + lane_width/2 + segment_3_length, n_poses_3).reshape((n_poses_3, 1))
        y_poses_3 = np.ones((x_poses_3.shape[0], 1))*self.map_height/2
        angles_3 = np.zeros((x_poses_3.shape[0], 1))

        # Merge these and return
        x_poses = np.vstack((x_poses_1, x_poses_2, x_poses_3))
        y_poses = np.vstack((y_poses_1, y_poses_2, y_poses_3))
        angles = np.vstack((angles_1, angles_2, angles_3))
        out = np.hstack((x_poses, y_poses, angles))
        return out

    def calc_offset(self, poses, curr_pose):  # poses could be stored in the class
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
        opt_pose = self.get_current_optimal_pose(poses)
        opt_x, opt_y, opt_heading = opt_pose

        # delta_psi
        delta_psi = heading - opt_heading

        # e
        segment = self.get_segment()
        assert segment in [1, 2, 3], "Unknown segment!"
        if segment == 1:
            e = x - opt_x
        elif segment == 2:
            circle_center = np.array([self.map_width / 2 + self.lane_width / 2, self.map_height/2 - self.lane_width/2])
            r = np.sqrt((x-circle_center[0])**2 + (y-circle_center[1])**2)
            r_opt = np.sqrt((opt_x-circle_center[0])**2 + (opt_y-circle_center[1])**2)
            e = r - r_opt
        elif segment == 3:
            e = y - opt_y

        return e, delta_psi
