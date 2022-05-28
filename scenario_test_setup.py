import sys
import numpy as np
import gym_carlo
import gym
import time
import argparse
import matplotlib.pyplot as plt
from gym_carlo.envs.interactive_controllers import GoalController
from utils_new import *
from mpc import MPC, Contigency_MPC
from nominal_trajectory import Nominal_Trajectory_Handler

from constants import MAP_WIDTH, MAP_HEIGHT, LANE_WIDTH, INITIAL_VELOCITY, DELTA_T, FIXED_CONTROL
from dynamics import calculate_x_y_pos


def controller_mapping(scenario_name, control):
    """Different scenarios have different number of goals, so let's just clip the user input -- also could be done via np.clip"""
    if control >= len(goals[scenario_name]):
        control = len(goals[scenario_name])-1
    return control


def take_action_placeholder(obs):
    return [0, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right", default="all")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--show_nominal", action="store_true", default=False)
    args = parser.parse_args()
    trajectory_handler = Nominal_Trajectory_Handler(map_height=MAP_HEIGHT, map_width=MAP_WIDTH, lane_width=LANE_WIDTH, velocity=INITIAL_VELOCITY, delta_t=DELTA_T)
    mpc_controller = MPC(pred_horizon=10, traj_handler=trajectory_handler)
    c_mpc = Contigency_MPC(pred_horizon=1, traj_handler=trajectory_handler)
    scenario_name = "intersection"

    if args.goal.lower() == 'all':
        goal_id = len(goals[scenario_name])
    else:
        goal_id = np.argwhere(np.array(goals[scenario_name]) == args.goal.lower())[0, 0]  # hmm, unreadable
    env = gym.make(scenario_name + 'Scenario-v0', goal=goal_id)

    trajectory_handler.reset_index()
    opt_traj = trajectory_handler.get_optimal_trajectory()
    trajectory_handler.increment_current_index()  # incr this once at the start. Now opt_pose will always be the pose where we are going

    # Debug - for checking the trajectory:
    if args.show_nominal:
        np.set_printoptions(threshold=sys.maxsize)
        print(opt_traj)
        plt.figure()
        x, y = opt_traj[:, 0], opt_traj[:, 1]
        plt.scatter(opt_traj[:, 0], opt_traj[:, 1], label="planned nominal trajectory")
        plt.xlim([0, MAP_WIDTH])
        plt.ylim([0, MAP_HEIGHT])
        plt.legend()
        plt.show()

    prev_state_traj = np.zeros((mpc_controller.sdim, mpc_controller.pred_horizon))  # pick first nominal trajectory as all 0s
    prev_controls = np.zeros((mpc_controller.adim, mpc_controller.pred_horizon))
    contigency_prev_state_traj = np.zeros((c_mpc.sdim, c_mpc.pred_horizon))
    contigency_prev_controls = np.zeros((c_mpc.adim, c_mpc.pred_horizon))

    # Arrays for logging
    u_vec = []
    state_vec = []
    x_y_heading_vec = []

    env.T = 200*env.dt - env.dt/2.  # Run for at most 200dt = 20 seconds

    env.seed(int(np.random.rand()*1e6))
    obs, done = env.reset(), False
    if args.visualize:
        env.render()
        interactive_policy = GoalController(env.world)

    # Visualize the optimal trajectory
    env.world.visualizer.draw_points(opt_traj[:, :2].T, color="chartreuse4")

    # Run the game until it is complete
    iteration = 0
    while not done:
        t = time.time()

        # Extract current state from controller
        U_y, r, e, delta_psi, curr_x, curr_y, curr_head = obs
        curr_state = np.array([U_y, r, e, delta_psi])
        print("current state:", curr_state)

        # Calculate the next control to take
        prev_controls, prev_state_traj = mpc_controller.calculate_control(curr_state, prev_state_traj, prev_controls)
        u0 = prev_controls[:, 0][0]

        # contingency_curr_state = np.concatenate((curr_state, curr_state))
        # contigency_prev_controls, contigency_prev_state_traj = c_mpc.calculate_control(contingency_curr_state, contigency_prev_state_traj, contigency_prev_controls)
        # u0 = prev_controls[:, 0][0]


        # For testing the dynamics
        # if iteration < 5:
        #     u0 = FIXED_CONTROL  # small negative seems to turn left, high positive also turns left
        # else:
        #     u0 = 0
        # u0 = (FIXED_CONTROL*iteration) / 100
        # if iteration > 100:
        #     u0 = FIXED_CONTROL
        print("Control u0:", u0)

        # Reformat the prev_states and prev_controls for next iterations linearization
        # Add the same control for the final timestep - correct?
        prev_controls = np.concatenate((prev_controls[:, 1:], prev_controls[:, -1].reshape(-1, 1)), axis=1)
        prev_state_traj = prev_state_traj[:, 1:]

        # Use the action in the environment
        action = [u0, 0]  # u0 as steering, 0 acceleration
        obs, _, done, _ = env.step(action)

        # Plot the planned trajectory in the x, y, heading - space
        # input_states = np.zeros((prev_state_traj.shape[0], prev_state_traj.shape[1]-1))
        # input_states[0:3, :] = prev_state_traj[0:3, :-1]  # Don't get last column
        # input_states[3, :] = prev_state_traj[3, 1:]
        opt_headings = np.zeros((prev_state_traj.shape[1]))  # input_states.shape[1]
        for i in range(prev_state_traj.shape[1]):
            _, _, opt_headings[i] = trajectory_handler.get_current_optimal_pose(opt_traj, i)
        proj_x, proj_y, proj_head = calculate_x_y_pos(env.ego.x, env.ego.y, env.ego.heading, opt_headings, trajectory_handler.get_U_x(), prev_state_traj)
        env.world.visualizer.draw_points(np.array([proj_x, proj_y]).squeeze(axis=2), color="crimson")

        # Log
        u_vec.append(u0)
        state_vec.append(curr_state)
        x_y_heading_vec.append([env.ego.x, env.ego.y, env.ego.heading])  # these will be offset by one compared to state and u

        iteration += 1
        trajectory_handler.increment_current_index()

        if args.visualize:
            env.render()
            while time.time() - t < env.dt*5/2:   # Temp * 10
                pass  # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
        if done:
            env.close()
            if args.visualize:
                time.sleep(1)
