import numpy as np
from .geometry import Point, Rectangle, Circle, Ring, Line
from typing import Union
import copy

from dynamics import true_dynamics
from nominal_trajectory import Nominal_Trajectory_Handler
from world_constants import MAP_WIDTH, MAP_HEIGHT, LANE_WIDTH, INITIAL_VELOCITY, DELTA_T


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

            # Other constants that we need TODO add these as input (in agents.py)
            # From Model Predictive Control for Vehicle Stabilization at the Limits of handling
            # by Craig Earl Beal and J. Christian Gerdes
            self.m = 1724  # Mass of vehicle
            self.Iz = 2600  # Yaw inertial of vehicle
            self.a = 1.35  # Distance to front axis from center of mass
            self.b = 1.15  # Distance to rear axis from center of mass

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
        # if friction:
        #     print("Friction is:", friction)
        # else:
        #     print("no fritcion")

        if self.movable:
            speed = self.speed
            heading = self.heading

            # Kinematic bicycle model dynamics based on
            # "Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design" by
            # Jason Kong, Mark Pfeiffer, Georg Schildbach, Francesco Borrelli
            # lr = self.rear_dist
            # lf = lr  # we assume the center of mass is the same as the geometric center of the entity
            # beta = np.arctan(lr / (lf + lr) * np.tan(self.inputSteering))
            #
            # new_angular_velocity = speed * self.inputSteering  # this is not needed and used for this model, but let's keep it for consistency (and to avoid if-else statements)
            # new_acceleration = self.inputAcceleration - self.friction
            # new_speed = np.clip(speed + new_acceleration * dt, self.min_speed, self.max_speed)
            # new_heading = heading + ((speed + new_speed)/lr)*np.sin(beta)*dt/2.
            # angle = (heading + new_heading)/2. + beta
            # new_center = self.center + (speed + new_speed)*Point(np.cos(angle), np.sin(angle))*dt / 2.
            # new_velocity = Point(new_speed * np.cos(new_heading), new_speed * np.sin(new_heading))

            """
            New approach
            """
            delta = self.inputSteering

            kappa = self.traj_handler.get_kappa()
            U_x = self.traj_handler.get_U_x()

            state = np.array([self.U_y, self.r, self.e, self.delta_psi])

            road_types = {
                0.25: "ice",
                0.95: "asphalt"
            }

            f_true = true_dynamics(state, delta, U_x, kappa, road_types[friction])
            U_y_dot, r_dot, e_dot, delta_psi_dot, s_dot = f_true

            # Update states
            self.U_y += U_y_dot * dt
            self.r += r_dot * dt
            self.e += e_dot * dt
            self.delta_psi += delta_psi_dot * dt
            s_prev = self.s
            self.s += s_dot * dt
            # update self.center


            '''
            # Point-mass dynamics based on
            # "Active Preference-Based Learning of Reward Functions" by
            # Dorsa Sadigh, Anca D. Dragan, S. Shankar Sastry, Sanjit A. Seshia

            new_angular_velocity = speed * self.inputSteering
            new_acceleration = self.inputAcceleration - self.friction * speed

            new_heading = heading + (self.angular_velocity + new_angular_velocity) * dt / 2.
            new_speed = np.clip(speed + (self.acceleration + new_acceleration) * dt / 2., self.min_speed, self.max_speed)

            new_velocity = Point(((speed + new_speed) / 2.) * np.cos((new_heading + heading) / 2.),
                                    ((speed + new_speed) / 2.) * np.sin((new_heading + heading) / 2.))

            new_center = self.center + (self.velocity + new_velocity) * dt / 2.

            '''

            opt_traj = self.traj_handler.get_optimal_trajectory()
            opt_x, opt_y, opt_heading = self.traj_handler.get_current_optimal_pose(opt_traj)
            # opt_x_next, opt_y_next, opt_heading_next = self.traj_handler.get_next_optimal_pose(opt_traj)
            
            s_cap, e_cap = np.array([np.cos(opt_heading), np.sin(opt_heading)]), np.array([-np.sin(opt_heading), np.cos(opt_heading)])

            actual_pos = np.array([opt_x, opt_y]) + (self.s - s_prev) * s_cap + self.e * e_cap
            actual_heading = np.mod(opt_heading + self.delta_psi, 2*np.pi)

            self.center = Point(actual_pos[0], actual_pos[1]) 
            self.heading = actual_heading
            # self.heading = np.mod(new_heading, 2*np.pi) # wrap the heading angle between 0 and +2pi
            # self.velocity = new_velocity
            # self.acceleration = new_acceleration
            # self.angular_velocity = new_angular_velocity

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
