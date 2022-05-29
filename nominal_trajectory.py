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
        self.unit_step = velocity * delta_t
        self.velocity = velocity

        # Calculate the different segments of the road
        unit_step = self.velocity * delta_t
        self.radius = 3*lane_width
        self.segment_1_length = map_height/2 - self.radius
        self.n_seg1 = np.arange(0, self.segment_1_length, self.unit_step).shape[0]
        self.segment_2_length = (2*self.radius*np.pi) / 4
        self.n_seg2 = int(self.segment_2_length / self.unit_step) + 1
        self.segment_3_length = map_width/2 - self.radius
        self.n_seg3 = np.arange(0, self.segment_3_length, self.unit_step).shape[0]
        self.circle_center = np.array([self.map_width / 2 - self.radius, self.map_height/2 - self.radius])

    def increment_current_index(self):
        """
        Should only be called once per iteration loop!!
        Increments the static index of the trajectory_handler
        """
        Nominal_Trajectory_Handler.current_index += 1

    def reset_index(self):
        Nominal_Trajectory_Handler.current_index = 0

    def get_segment(self, future_steps=0):
        """
        Returns segment 1,2 or 3 depending on the static index
        """
        # TODO think about edge cases(?)
        ind = Nominal_Trajectory_Handler.current_index+future_steps
        if ind < self.n_seg1:
            return 1
        elif self.n_seg1 <= ind < self.n_seg1+self.n_seg2:
            return 2
        else:
            return 3

    def get_kappa(self, future_steps):
        seg = self.get_segment(future_steps)
        if seg == 1 or seg == 3:
            return 0
        else:
            return 1/self.radius  # kappa = 1/r

    def get_U_x(self):
        """
        U_x = v which is constant across the trajectory
        """
        return self.velocity

    def get_s(self):
        return self.unit_step * Nominal_Trajectory_Handler.current_index

    def get_current_optimal_pose(self, opt_traj, future_steps=0):
        """
        Given that the car is in state s_t, this should always be giving the optimal state for s_{t+1}
        So if we want to compare x and y that should be done after incrementing the current dynamics to the next state
        """
        return opt_traj[Nominal_Trajectory_Handler.current_index+future_steps, :]


    def get_optimal_trajectory(self):
        """
        Generates the optimal (x, y, angle) poses across a trajectory
        as a n x 3 np array

        # TODO:
        - handle edge cases between the segments
        - look at if n_poses_i isnt an integer for any segment
        """
        ### First segment, straight ###
        y_poses_1 = np.arange(0, self.segment_1_length, self.unit_step).reshape((-1, 1))
        x_poses_1 = np.ones((y_poses_1.shape[0], 1))*self.map_width/2
        angles_1 = np.ones((y_poses_1.shape[0], 1))*(np.pi/2)

        ### Second segment, circle ###
        x_poses_2 = []
        y_poses_2 = []
        angles_2 = []
        for i in range(self.n_seg2):
            # Calculate poses across the arc
            pos_ang = ((np.pi/2) * i /self.n_seg2)
            x_poses_2.append( self.circle_center[0] + self.radius*np.cos(pos_ang) )
            y_poses_2.append( self.circle_center[1] + self.radius*np.sin(pos_ang) )

            # Calculate heading angles across the arc
            steer_ang = (np.pi/2) + ((np.pi/2) * i / self.n_seg2)
            angles_2.append(steer_ang)

        x_poses_2 = np.array(x_poses_2).reshape((self.n_seg2, 1))
        y_poses_2 = np.array(y_poses_2).reshape((self.n_seg2, 1))
        angles_2 = np.array(angles_2).reshape((self.n_seg2, 1))

        ### Third segment, straight right ###
        x_poses_3 = np.arange(self.segment_3_length, 0, -self.unit_step).reshape(-1, 1)
        y_poses_3 = np.ones((x_poses_3.shape[0], 1))*self.map_height/2
        angles_3 = np.ones((x_poses_3.shape[0], 1))*np.pi

        # Merge these and return
        x_poses = np.vstack((x_poses_1, x_poses_2, x_poses_3))
        y_poses = np.vstack((y_poses_1, y_poses_2, y_poses_3))
        angles = np.vstack((angles_1, angles_2, angles_3))
        out = np.hstack((x_poses, y_poses, angles))

        # ### RIGHT HAND TURN ###
        # ### First segment, straight ###
        # y_poses_1 = np.arange(0, self.segment_1_length, self.unit_step).reshape((-1, 1))
        # x_poses_1 = np.ones((y_poses_1.shape[0], 1))*self.map_width/2
        # angles_1 = np.ones((y_poses_1.shape[0], 1))*(np.pi/2)
        # # s = np.copy(y_poses_1)
        #
        # ### Second segment, circle ###
        # circle_center = np.array([self.map_width/2 + self.lane_width/2, self.map_height/2 - self.lane_width/2])  # from geometry of our right turn
        # r = self.lane_width/2
        # # s_current = s[-1]  # the distance travelled after segment 1
        # x_poses_2 = []
        # y_poses_2 = []
        # angles_2 = []
        # for i in range(self.n_seg2):
        #     # Calculate poses across the arc
        #     pos_ang = np.pi - ((np.pi/2) * i /self.n_seg2)
        #     x_poses_2.append( circle_center[0] + r*np.cos(pos_ang) )
        #     y_poses_2.append( circle_center[1] + r*np.sin(pos_ang) )
        #
        #     # Calculate heading angles across the arc
        #     steer_ang = (np.pi/2) - ((np.pi/2) * i / self.n_seg2)
        #     angles_2.append(steer_ang)
        #
        #     # Build s
        #     # circ_arc_len = ((np.pi/2) /self.n_seg2) * r  # theta * r
        #     # s_current += circ_arc_len
        #     # s = np.append(s, s_current)
        #
        # x_poses_2 = np.array(x_poses_2).reshape((self.n_seg2, 1))
        # y_poses_2 = np.array(y_poses_2).reshape((self.n_seg2, 1))
        # angles_2 = np.array(angles_2).reshape((self.n_seg2, 1))
        #
        # ### Third segment, straight right ###
        # x_poses_3 = np.arange(self.map_width/2 + self.lane_width/2, self.map_width/2 + self.lane_width/2 + self.segment_3_length, self.unit_step).reshape(-1, 1)  # np.linspace(map_width/2 + lane_width/2, map_width/2 + lane_width/2 + segment_3_length, n_poses_3).reshape((n_poses_3, 1))
        # y_poses_3 = np.ones((x_poses_3.shape[0], 1))*self.map_height/2
        # angles_3 = np.zeros((x_poses_3.shape[0], 1))
        #
        # # Get final s and story it in the class
        # # steps_3 = np.arange(0, self.segment_3_length, self.unit_step)
        # # s = np.concatenate((s, np.add.accumulate(steps_3) + s_current))
        # # self.s = s
        #
        # # Merge these and return
        # x_poses = np.vstack((x_poses_1, x_poses_2, x_poses_3))
        # y_poses = np.vstack((y_poses_1, y_poses_2, y_poses_3))
        # angles = np.vstack((angles_1, angles_2, angles_3))
        # out = np.hstack((x_poses, y_poses, angles))

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
            r = np.sqrt((x-self.circle_center[0])**2 + (y-self.circle_center[1])**2)
            r_opt = np.sqrt((opt_x-self.circle_center[0])**2 + (opt_y-self.circle_center[1])**2)
            e = r - r_opt
        elif segment == 3:
            e = y - opt_y

        return e, delta_psi
