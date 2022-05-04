import numpy as np
import os
from gym_carlo.envs.geometry import Point

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