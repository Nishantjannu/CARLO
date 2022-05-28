# Important settings for the map
MAP_WIDTH = 200  # 80
MAP_HEIGHT = 100  # 120
LANE_WIDTH = 16  # 4.4
INITIAL_VELOCITY = 10  # 3
DELTA_T = 0.1
SIDEWALK_WIDTH = 2.0
LANE_MARKER_HEIGHT = 3.8
LANE_MARKER_WIDTH = 0.5
BUILDING_WIDTH = (MAP_WIDTH - 2*SIDEWALK_WIDTH - 2*LANE_WIDTH - LANE_MARKER_WIDTH) / 2.
TOP_BUILDING_HEIGHT = MAP_HEIGHT - (LANE_MARKER_WIDTH/2. + LANE_WIDTH + SIDEWALK_WIDTH)  # intersection_y will be subtracted
BOTTOM_BUILDING_HEIGHT = -LANE_MARKER_WIDTH/2. - LANE_WIDTH - SIDEWALK_WIDTH  # intersection_y will be added

# Parameters for the car
# From Model Predictive Control for Vehicle Stabilization at the Limits of handling
# by Craig Earl Beal and J. Christian Gerdes
CAR_MASS = 1724  # Mass of vehicle
CAR_YAW_INERTIAL = 2600  # Yaw inertial of vehicle
CAR_FRONT_AXIS_DIST = 1.35  # Distance to front axis from center of mass
CAR_BACK_AXIS_DIST = 1.15  # Distance to rear axis from center of mass

FIXED_CONTROL = 1
DEBUG_LINERIZED_DYNAMICS = 1
