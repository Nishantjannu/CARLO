import sys
import numpy as np
import gym_carlo
import gym
import time
import argparse
import matplotlib.pyplot as plt
from gym_carlo.envs.interactive_controllers import GoalController
from utils_new import *
from mpc import MPC
from nominal_trajectory import Nominal_Trajectory_Handler

from constants import MAP_WIDTH, MAP_HEIGHT, LANE_WIDTH, INITIAL_VELOCITY, DELTA_T


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
    trajectory_handler = Nominal_Trajectory_Handler(map_height=MAP_HEIGHT, map_width=MAP_WIDTH, lane_width=LANE_WIDTH, velocity=INITIAL_VELOCITY, delta_t=DELTA_T)
    mpc_controller = MPC(pred_horizon=20, traj_handler=trajectory_handler)
    scenario_name = "intersection"

    if args.goal.lower() == 'all':
        goal_id = len(goals[scenario_name])
    else:
        goal_id = np.argwhere(np.array(goals[scenario_name]) == args.goal.lower())[0, 0]  # hmm, unreadable
    env = gym.make(scenario_name + 'Scenario-v0', goal=goal_id)

    trajectory_handler.reset_index()
    opt_traj = trajectory_handler.get_optimal_trajectory()

    plt.figure()
    x, y = opt_traj[:, 0], opt_traj[:, 1]
    # plt.scatter(opt_traj[:, 0], opt_traj[:, 1], label="planned nominal trajectory")
    # plt.xlim([0, MAP_WIDTH])
    # plt.ylim([0, MAP_HEIGHT])
    # plt.legend()
    # plt.show()
    # Debug:
    # np.set_printoptions(threshold=sys.maxsize)
    # print(opt_traj)
    prev_state_traj = np.zeros((mpc_controller.sdim, mpc_controller.pred_horizon))  # pick first nominal trajectory as all 0s
    prev_controls = np.zeros((mpc_controller.adim, mpc_controller.pred_horizon))

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

            #print("s", trajectory_handler.get_s())

            # e, delta_psi = trajectory_handler.calc_offset(opt_traj, curr_pos)

            U_y, r, e, delta_psi, curr_x, curr_y, curr_head = obs

            # action = simple_controller_1(delta_psi)
            # action = take_action_placeholder(obs)
            # Input to MPC-controler: [U_y, r, delta_psi, e]
            curr_state = np.array([U_y, r, e, delta_psi])
            print("current state:", curr_state)

            # prev_controls, prev_state_traj = mpc_controller.calculate_control(curr_state, prev_state_traj, prev_controls)
            # u0 = prev_controls[:, 0][0]
            u0 = 0
            print("Control u0:", u0)
            #print("u0:", u0)
            action = [u0, 0]  # u0 as steering, 0 acceleration
            obs, _, done, _ = env.step(action)

            # print("opt_traj[:, 0:2].shape", opt_traj[:, 0:2].shape)
            # env.world.visualizer.draw_points(opt.traj[:, 0:2])

            # plt.subplot(2, 1, 1)
            # plt.plot(prev_state_traj[:, 0], label="planned U_y")
            # plt.plot(prev_state_traj[:, 1], label="planned r")
            # plt.plot(prev_state_traj[:, 2], label="planned e")
            # plt.plot(prev_state_traj[:, 3], label="planned delta_psi")
            # plt.legend()
            # plt.subplot(2, 1, 2)
            # print("xshape", x.shape, "yshape", y.shape)
            # plt.plot(prev_controls, "-o", label="Controls over the horizon")
            # plt.legend()
            # plt.show()

            trajectory_handler.increment_current_index()

            if args.visualize:
                env.render()
                while time.time() - t < env.dt/2:   # Temp * 10
                    pass  # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                env.close()
                if args.visualize:
                    time.sleep(1)
                if env.target_reached:
                    success_counter += 1
    if not args.visualize:
        print('Success Rate = ' + str(float(success_counter)/episode_number))
