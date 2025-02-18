import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

from .world import World
from .agents import Car, RectangleBuilding, Pedestrian, Painting, IcePatch, Asphalt
from .geometry import Point
from .graphics import Text, Point as pnt  # very unfortunate indeed
from constants import *

PPM = 5  # 5  # pixels per meter

class IntersectionScenario(gym.Env):
    def __init__(self, goal):
        assert 0 <= goal <= 3, 'Undefined goal'

        self.seed(0)  # just in case we forget seeding

        self.active_goal = goal

        self.init_ego = Car(Point(MAP_WIDTH/2, 0), heading=np.pi/2)  # Car(Point(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2. + LANE_WIDTH/2., 0), heading=np.pi/2)
        self.init_ego.velocity = Point(1., 0.)
        self.init_ego.min_speed = 0.
        self.init_ego.max_speed = 20.

        self.dt = DELTA_T
        self.T = 200  # Maximum length of simulation

        self.reset()

    def reset(self):
        self.world = World(self.dt, width=MAP_WIDTH, height=MAP_HEIGHT, ppm=PPM)

        self.ego = self.init_ego.copy()

        # Randomize intersection location, start position and start velocity IMPORTANT TO KNOW
        self.intersection_y = MAP_HEIGHT/2  # self.np_random.rand()*MAP_HEIGHT/2. + MAP_HEIGHT/4.
        self.ego.velocity = Point(0, INITIAL_VELOCITY)

        self.targets = []
        self.targets.append(Point(0., self.intersection_y + LANE_WIDTH/2. + LANE_MARKER_WIDTH/2.))  # Might want to remove these
        self.targets.append(Point(MAP_WIDTH/2. + LANE_WIDTH/2. + LANE_MARKER_WIDTH/2., MAP_HEIGHT))
        self.targets.append(Point(MAP_WIDTH, self.intersection_y - LANE_WIDTH/2. - LANE_MARKER_WIDTH/2.))

        # Ice patch location
        icex1 = MAP_WIDTH / 2  # center x
        icey1 = self.intersection_y  # center y
        icewidth = 40  # width
        iceheigth = 40  # height

        # Asphalt (drawing it here hides it)
        height1 = MAP_HEIGHT - (icey1 + iceheigth / 2)
        self.world.add(
        Asphalt(Point(icex1, icey1 + iceheigth / 2 + height1 / 2), Point(MAP_WIDTH, height1)))  # Upper segment
        height2 = icey1 - iceheigth / 2
        self.world.add(Asphalt(Point(icex1, height2 / 2), Point(MAP_WIDTH, height2)))  # Lower
        width1 = icex1 - icewidth / 2
        self.world.add(Asphalt(Point(width1 / 2, icey1), Point(width1, MAP_HEIGHT)))  # Left
        width2 = MAP_WIDTH - (icex1 + icewidth / 2)
        self.world.add(Asphalt(Point(icex1 + icewidth / 2 + width2 / 2, icey1), Point(width2, MAP_HEIGHT)))  # Right

        # Ice patch
        # self.world.add(Asphalt(Point(icex1, icey1), Point(icewidth, iceheigth)))
        self.world.add(IcePatch(Point(icex1, icey1), Point(icewidth, iceheigth)))

        # Pretty things on the side of the road
        self.world.add(Painting(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))
        self.world.add(Painting(Point(BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))
        self.world.add(Painting(Point(BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))
        self.world.add(Painting(Point(MAP_WIDTH - BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH+2*SIDEWALK_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y+2*SIDEWALK_WIDTH), 'gray64'))

        # lane markers on the bottom road
        for y in np.arange(LANE_MARKER_HEIGHT/2., self.intersection_y - LANE_MARKER_WIDTH/2 - LANE_WIDTH - LANE_MARKER_HEIGHT/2, 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(MAP_WIDTH/2., y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        # lane markers on the right road
        for x in np.arange(MAP_WIDTH/2. + LANE_MARKER_WIDTH/2 + LANE_WIDTH + LANE_MARKER_HEIGHT/2, MAP_WIDTH - LANE_MARKER_HEIGHT/2., 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(x, self.intersection_y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white', heading = np.pi/2))
        # lane markers on the top road
        for y in np.arange(self.intersection_y + LANE_MARKER_WIDTH/2 + LANE_WIDTH + LANE_MARKER_HEIGHT/2, MAP_HEIGHT - LANE_MARKER_HEIGHT/2., 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(MAP_WIDTH/2., y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white'))
        # lane markers on the left road
        for x in np.arange(LANE_MARKER_HEIGHT/2, MAP_WIDTH/2. - LANE_MARKER_WIDTH/2 - LANE_WIDTH - LANE_MARKER_HEIGHT/2, 2*LANE_MARKER_HEIGHT):
            self.world.add(Painting(Point(x, self.intersection_y), Point(LANE_MARKER_WIDTH, LANE_MARKER_HEIGHT), 'white', heading = np.pi/2))

        # Pretty rectangles in the corners of the world
        self.world.add(RectangleBuilding(Point(MAP_WIDTH - BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y)))
        self.world.add(RectangleBuilding(Point(BUILDING_WIDTH/2., MAP_HEIGHT - (TOP_BUILDING_HEIGHT-self.intersection_y)/2.), Point(BUILDING_WIDTH, TOP_BUILDING_HEIGHT-self.intersection_y)))
        self.world.add(RectangleBuilding(Point(BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y)))
        self.world.add(RectangleBuilding(Point(MAP_WIDTH - BUILDING_WIDTH/2., (BOTTOM_BUILDING_HEIGHT+self.intersection_y)/2.), Point(BUILDING_WIDTH, BOTTOM_BUILDING_HEIGHT+self.intersection_y)))

        self.world.add(self.ego)

        return self._get_obs()

    def close(self):
        self.world.close()

    # @property
    # def observation_space(self):
    #     low = np.array([0, 0, self.ego.min_speed, 0, MAP_HEIGHT/4.])
    #     high = np.array([MAP_WIDTH, MAP_HEIGHT, self.ego.max_speed, 2*np.pi, MAP_HEIGHT*3./4.])
    #     return Box(low=low, high=high)

    @property
    def action_space(self):
        return Box(low=np.array([-0.5,-2.0]), high=np.array([0.5,1.5]))

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def target_reached(self):
        if self.active_goal < len(self.targets):
            return self.targets[self.active_goal].distanceTo(self.ego) < 1.
        return np.min([self.targets[i].distanceTo(self.ego) for i in range(len(self.targets))]) < 1.

    @property
    def collision_exists(self):
        return self.world.collision_exists()

    @property
    def ego_approaching_intersection(self):
        return MAP_HEIGHT - (TOP_BUILDING_HEIGHT - self.intersection_y) - SIDEWALK_WIDTH - LANE_WIDTH > self.ego.y > BOTTOM_BUILDING_HEIGHT + self.intersection_y - 5 and \
                BUILDING_WIDTH < self.ego.x < MAP_WIDTH - BUILDING_WIDTH

    def step(self, action):
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self.ego.set_control(action[0], action[1])
        self.world.tick()

        return self._get_obs(), self._get_reward(), self.target_reached or self.world.t >= self.T, {}  # self.collision_exists or

    def _get_reward(self):
        """
        The reward for a given state. Should be updated. Could also be removed since we wont do reinforcement learning
        """
        if self.collision_exists:
            return -200
        if self.active_goal < len(self.targets):
            return -0.01*self.targets[self.active_goal].distanceTo(self.ego)
        return -0.01*np.min([self.targets[i].distanceTo(self.ego) for i in range(len(self.targets))])

    def _get_obs(self):
        """
        Returns observation:

        """
        return np.array([self.ego.U_y, self.ego.r, self.ego.e, self.ego.delta_psi, self.ego.center.x, self.ego.center.y, self.ego.heading])

    def render(self, mode='rgb'):
        self.world.render()

    def write(self, text):  # this is hacky, it would be good to have a write() function in world class
        if hasattr(self, 'txt'):
            self.txt.undraw()
        self.txt = Text(pnt(PPM*(MAP_WIDTH - BUILDING_WIDTH+2), self.world.visualizer.display_height - PPM*10), text)
        self.txt.draw(self.world.visualizer.win)
