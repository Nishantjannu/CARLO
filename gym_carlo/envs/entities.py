import numpy as np
from .geometry import Point, Rectangle, Circle, Ring, Line
from typing import Union
import copy

from dynamics import true_dynamics, linear_dynamics, calculate_x_y_pos
from nominal_trajectory import Nominal_Trajectory_Handler
from constants import *


class Entity:
    def __init__(self, center: Point, heading: float, movable: bool = True, friction: float = 0):
        self.center = center  # this is x, y
        self.heading = heading
        self.movable = movable
        self.color = 'ghost white'
        self.collidable = True
        self.friction = friction  # Moved this out
        if movable:
            self.velocity = Point(0, 0)  # this is xp, yp
            self.acceleration = 0  # this is vp (or speedp)
            self.angular_velocity = 0  # this is headingp
            self.inputSteering = 0
            self.inputAcceleration = 0
            self.max_speed = np.inf
            self.min_speed = 0

            # Our Bicycle model
            self.U_y = 0
            self.r = 0
            self.e = 0
            self.delta_psi = 0
            self.s = 0

            self.prev_control = 0

            # Other constants that we need TODO add these as input (in agents.py)
            # From Model Predictive Control for Vehicle Stabilization at the Limits of handling
            # by Craig Earl Beal and J. Christian Gerdes
            self.m = CAR_MASS  # Mass of vehicle
            self.Iz = CAR_YAW_INERTIAL  # Yaw inertial of vehicle
            self.a = CAR_FRONT_AXIS_DIST  # Distance to front axis from center of mass
            self.b = CAR_BACK_AXIS_DIST  # Distance to rear axis from center of mass

            self.traj_handler = Nominal_Trajectory_Handler(map_height=MAP_HEIGHT, map_width=MAP_WIDTH, lane_width=LANE_WIDTH, velocity=INITIAL_VELOCITY, delta_t=DELTA_T)

    @property
    def speed(self) -> float:
        return self.velocity.norm(p = 2) if self.movable else 0

    def set_control(self, inputSteering: float, inputAcceleration: float):
        self.inputSteering = inputSteering
        self.inputAcceleration = inputAcceleration

    @property
    def rear_dist(self) -> float:  # distance between the rear wheels and the center of mass. This is needed to implement the kinematic bicycle model dynamics
        if isinstance(self, RectangleEntity):
            # only for this function, we assume
            # (i) the longer side of the rectangle is always the nominal direction of the car
            # (ii) the center of mass is the same as the geometric center of the RectangleEntity.
            return np.maximum(self.size.x, self.size.y) / 2.
        elif isinstance(self, CircleEntity):
            return self.radius
        elif isinstance(self, RingEntity):
            return (self.inner_radius + self.outer_radius) / 2.
        raise NotImplementedError

    def tick(self, dt, friction=None):
        if self.movable:
            speed = self.speed
            heading = self.heading
            delta = self.inputSteering

            kappa = self.traj_handler.get_kappa(0)
            U_x = self.traj_handler.get_U_x()
            state = np.array([self.U_y, self.r, self.e, self.delta_psi])

            # This should be changed, should get material directly rather than doing this conversion
            road_types = {
                0.25: "ice",
                0.95: "asphalt"
            }

            # Get current optimal poses
            opt_traj = self.traj_handler.get_optimal_trajectory()
            opt_x, opt_y, opt_heading = self.traj_handler.get_current_optimal_pose(opt_traj)

            if DEBUG_LINERIZED_DYNAMICS == 1:
                prev_vals = {
                    "state": state,
                    "control": self.prev_control
                }
                A, B, C = linear_dynamics(prev_vals, U_x, kappa, road_types[friction])
                new_state = state + dt*A@state + dt*B*delta + dt*C
                self.U_y = new_state[0]
                self.r = new_state[1]
                self.e = new_state[2]
                self.delta_psi = new_state[3]

                self.prev_control = delta

            else:
                # Calculate current dynamics
                f_true = true_dynamics(state, delta, U_x, kappa, road_types[friction])
                U_y_dot, r_dot, e_dot, delta_psi_dot, s_dot = f_true

                # Update states
                self.U_y += U_y_dot * dt
                self.r += r_dot * dt
                self.e += e_dot * dt
                self.delta_psi += delta_psi_dot * dt
                self.s += s_dot * dt

            # Find the new x, y and z of the car
            ipt_state = state.copy()
            # ipt_state[3] = self.delta_psi  # Update this value to shift it
            ipt_state = ipt_state.reshape((ipt_state.shape[0], 1))
            new_car_state = calculate_x_y_pos(self.center.x, self.center.y, self.heading, [opt_heading], U_x, ipt_state)

            # Update the car's x, y and heading
            self.center = Point(new_car_state[0], new_car_state[1])
            self.heading = new_car_state[2][0][0]  # - problem with arrays

            self.buildGeometry()

    def isInside(self, other: Union['Line', 'Rectangle', 'Circle', 'Ring']) -> bool:
        if isinstance(other, Line):
            AM = Line(other.p1, self)
            BM = Line(self, other.p2)
            return np.close(np.abs(AM.dot(BM)), AM.length * BM.length)

        elif isinstance(other, Rectangle):
            # Based on https://stackoverflow.com/a/2763387
            AB = Line(other.c1, other.c2)
            AM = Line(other.c1, self)
            BC = Line(other.c2, other.c3)
            BM = Line(other.c2, self)

            return 0 <= AB.dot(AM) <= AB.dot(AB) and 0 <= BC.dot(BM) <= BC.dot(BC)

        elif isinstance(other, Circle):
            return self.distanceTo(other.m) <= other.r

        elif isinstance(other, Ring):
            return other.r_inner <= self.distanceTo(other.m) <= other.r_outer

        raise NotImplementedError

    def buildGeometry(self):  # builds the obj
        raise NotImplementedError

    def collidesWith(self, other: Union['Point','Entity']) -> bool:
        if isinstance(other, Entity):
            return self.obj.intersectsWith(other.obj)
        elif isinstance(other, Point):
            return self.obj.intersectsWith(other)
        raise NotImplementedError

    def distanceTo(self, other: Union['Point','Entity']) -> float:
        if isinstance(other, Entity):
            return self.obj.distanceTo(other.obj)
        elif isinstance(other, Point):
            return self.obj.distanceTo(other)
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y

    @property
    def xp(self):
        return self.velocity.x

    @property
    def yp(self):
        return self.velocity.y

class RectangleEntity(Entity):
    def __init__(self, center: Point, heading: float, size: Point, movable: bool = True, friction: float = 0):
        super(RectangleEntity, self).__init__(center, heading, movable, friction)
        self.size = size
        self.buildGeometry()

    @property
    def edge_centers(self):
        edge_centers = np.zeros((4,2), dtype=np.float32)
        x = self.center.x
        y = self.center.y
        w = self.size.x
        h = self.size.y
        edge_centers[0] = [x + w / 2. * np.cos(self.heading), y + w / 2. * np.sin(self.heading)]
        edge_centers[1] = [x - h / 2. * np.sin(self.heading), y + h / 2. * np.cos(self.heading)]
        edge_centers[2] = [x - w / 2. * np.cos(self.heading), y - w / 2. * np.sin(self.heading)]
        edge_centers[3] = [x + h / 2. * np.sin(self.heading), y - h / 2. * np.cos(self.heading)]
        return edge_centers

    @property
    def corners(self):
        ec = self.edge_centers
        c = np.array([self.center.x, self.center.y])
        corners = []
        corners.append(Point(*(ec[1] + ec[0] - c)))
        corners.append(Point(*(ec[2] + ec[1] - c)))
        corners.append(Point(*(ec[3] + ec[2] - c)))
        corners.append(Point(*(ec[0] + ec[3] - c)))
        return corners

    def buildGeometry(self):
        C = self.corners
        self.obj = Rectangle(*C[:-1])

class CircleEntity(Entity):
    def __init__(self, center: Point, heading: float, radius: float, movable: bool = True, friction: float = 0):
        super(CircleEntity, self).__init__(center, heading, movable, friction)
        self.radius = radius
        self.buildGeometry()

    def buildGeometry(self):
        self.obj = Circle(self.center, self.radius)

class RingEntity(Entity):
    def __init__(self, center: Point, heading: float, inner_radius: float, outer_radius: float, movable: bool = True, friction: float = 0):
        super(RingEntity, self).__init__(center, heading, movable, friction)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.buildGeometry()

    def buildGeometry(self):
        self.obj = Ring(self.center, self.inner_radius, self.outer_radius)
