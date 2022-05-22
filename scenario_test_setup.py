import sys
import numpy as np
import gym_carlo
import gym
import time
import argparse
from gym_carlo.envs.interactive_controllers import GoalController
from utils_new import *
from mpc import MPC

# TODO change the way these are accessed/set
from gym_carlo.envs.intersection_scenario import MAP_WIDTH, MAP_HEIGHT, LANE_WIDTH, INITIAL_VELOCITY, DELTA_T


def controller_mapping(scenario_name, control):
    """Different scenarios have different number of goals, so let's just clip the user input -- also could be done via np.clip"""
    if control >= len(goals[scenario_name]):
        control = len(goals[scenario_name])-1
    return control


def take_action_placeholder(obs):
    return [0, 0]


def simple_controller_1(delta_psi):
    if delta_psi > 0:
        steering = -1
    elif delta_psi < 0:
        steering = 1
    else:
        steering = 0
    return [steering, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right", default="all")
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    controller = MPC()
    scenario_name = "intersection"

    if args.goal.lower() == 'all':
        goal_id = len(goals[scenario_name])
    else:
        goal_id = np.argwhere(np.array(goals[scenario_name]) == args.goal.lower())[0, 0]  # hmm, unreadable
    env = gym.make(scenario_name + 'Scenario-v0', goal=goal_id)


    opt_traj = optimal_trajectory(MAP_HEIGHT, MAP_WIDTH, LANE_WIDTH, INITIAL_VELOCITY, DELTA_T)
    # Debug
    # np.set_printoptions(threshold=sys.maxsize)
    # print(opt_traj)

    episode_number = 10 if args.visualize else 100
    success_counter = 0
    env.T = 200*env.dt - env.dt/2.  # Run for at most 200dt = 20 seconds

    # Run for a number of episodes
    for _ in range(episode_number):
        env.seed(int(np.random.rand()*1e6))
        obs, done = env.reset(), False
        if args.visualize:
            env.render()
            interactive_policy = GoalController(env.world)

        # Run the game until it is complete
        iteration = 0
        while not done:
            t = time.time()
            # obs = np.array(obs).reshape(1, -1)
            u = controller_mapping(scenario_name, interactive_policy.control) if args.visualize else goal_id  # What does this do..?

            print("obs, obs.shape:", obs, obs.shape)
            x, y, heading = obs[0], obs[1], obs[3]
            curr_pos = (x, y, heading)
            e, delta_psi = calc_offset(opt_traj, curr_pos, MAP_HEIGHT, MAP_WIDTH, LANE_WIDTH, INITIAL_VELOCITY, DELTA_T, iteration)
            action = simple_controller_1(delta_psi)
            print("delta_psi:", delta_psi, "action", action)

            # action = take_action_placeholder(obs)
            # action, _ = controller.calculate_control(obs)
            obs, _, done, _ = env.step(action)


            iteration += 1

            if args.visualize:
                env.render()
                while time.time() - t < env.dt/2:
                    pass  # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                env.close()
                if args.visualize:
                    time.sleep(1)
                if env.target_reached:
                    success_counter += 1
    if not args.visualize:
        print('Success Rate = ' + str(float(success_counter)/episode_number))
