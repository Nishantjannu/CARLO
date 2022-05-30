import sys
import numpy as np
import gym_carlo
import gym
import time
import argparse
import matplotlib.pyplot as plt
from gym_carlo.envs.interactive_controllers import GoalController
from utils_new import *
from mpc import MPC, Contigency_MPC, MPC_ice
from nominal_trajectory import Nominal_Trajectory_Handler

from constants import MAP_WIDTH, MAP_HEIGHT, LANE_WIDTH, INITIAL_VELOCITY, DELTA_T, FIXED_CONTROL
from constants import CAR_FRONT_AXIS_DIST as a, CAR_BACK_AXIS_DIST as b
from dynamics import *


def plot_state_trajectories(states, title):
    fig, axes = plt.subplots(nrows=4, ncols=1)
    fig.suptitle(title)
    axes[0].plot(states[0, :], label="U_y")
    axes[0].legend()
    axes[1].plot(states[1, :], label="r")
    axes[1].legend()
    axes[2].plot(states[2, :], label="e")
    axes[2].legend()
    axes[3].plot(states[3, :], label="Delta_psi")
    axes[3].legend()

def plot_control(controls):
    plt.figure()
    plt.plot(controls)
    plt.xlabel("Time")
    plt.ylabel("Delta")



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
    parser.add_argument("--use_cont", action="store_true", default=False)
    parser.add_argument("--use_ice", action="store_true", default=False)
    args = parser.parse_args()
    trajectory_handler = Nominal_Trajectory_Handler(map_height=MAP_HEIGHT, map_width=MAP_WIDTH, lane_width=LANE_WIDTH, velocity=INITIAL_VELOCITY, delta_t=DELTA_T)
    prediction_horizon = 100  # 50 for cmpc   # 45, 100
    if not args.use_cont:
        if args.use_ice:
            mpc_controller = MPC_ice(pred_horizon=prediction_horizon, traj_handler=trajectory_handler)
        else:
            mpc_controller = MPC(pred_horizon=prediction_horizon, traj_handler=trajectory_handler)
    else:
        c_mpc = Contigency_MPC(pred_horizon=prediction_horizon, traj_handler=trajectory_handler)
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

    if not args.use_cont:
        prev_state_traj = np.zeros((mpc_controller.sdim, mpc_controller.pred_horizon))  # pick first nominal trajectory as all 0s
        prev_controls = np.zeros((mpc_controller.adim, mpc_controller.pred_horizon))
    else:
        contigency_prev_state_traj = np.zeros((c_mpc.sdim, c_mpc.pred_horizon))
        contigency_prev_controls = np.zeros((c_mpc.adim, c_mpc.pred_horizon))
    u0 = 0  # init this value

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
        #
        if not args.use_cont:
            alpha_limit = 25
            Ux = trajectory_handler.get_U_x()
            alpha_f_act = (180/np.pi)*(prev_state_traj[0,-1] + a*prev_state_traj[1,-1])/Ux - prev_controls[0,-1]
            alpha_f_act = (180/np.pi)*(prev_state_traj[0,-1] - b*prev_state_traj[1,-1])/Ux
            if np.abs(alpha_f_act) > alpha_limit or np.abs(alpha_f_act) > alpha_limit:
                print("final alphas above limit of:", alpha_limit, "selecting nominal trajectory to linearize around")
                prev_controls = np.zeros_like(prev_controls)
                prev_state_traj = np.zeros_like(prev_state_traj)
            prev_controls, prev_state_traj = mpc_controller.calculate_control(curr_state, u0, np.zeros((mpc_controller.sdim, mpc_controller.pred_horizon)), np.zeros((mpc_controller.adim, mpc_controller.pred_horizon)))
            u0 = prev_controls[:, 0][0]

            if iteration == 0:
                initial_plan = prev_state_traj

        if args.use_cont:
            alpha_limit = 25
            Ux = trajectory_handler.get_U_x()
            alpha_f_act = (180/np.pi)*(contigency_prev_state_traj[0,-1] + a*contigency_prev_state_traj[1,-1])/Ux - contigency_prev_controls[0,-1]
            alpha_f_act = (180/np.pi)*(contigency_prev_state_traj[0,-1] - b*contigency_prev_state_traj[1,-1])/Ux
            if np.abs(alpha_f_act) > alpha_limit or np.abs(alpha_f_act) > alpha_limit:
                print("final alphas above limit of:", alpha_limit, "selecting nominal trajectory to linearize around")
                contigency_prev_controls = np.zeros_like(contigency_prev_controls)
                contigency_prev_state_traj = np.zeros_like(contigency_prev_state_traj)

            curr_state_c = np.concatenate((curr_state, curr_state))
            u0_c = np.array([u0, u0])
            contigency_prev_controls, contigency_prev_state_traj, status = c_mpc.calculate_control(curr_state_c, u0_c, contigency_prev_state_traj, contigency_prev_controls)
            u0 = contigency_prev_controls[:, 0][0]

            if iteration == 0:
                initial_plan = contigency_prev_state_traj


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
        if not args.use_cont:
            prev_controls = np.concatenate((prev_controls[:, 1:], prev_controls[:, -1].reshape(-1, 1)), axis=1)
            prev_state_traj = prev_state_traj[:, 1:]
        else:
            contigency_prev_controls = np.concatenate((contigency_prev_controls[:, 1:], contigency_prev_controls[:, -1].reshape(-1, 1)), axis=1)
            contigency_prev_state_traj = contigency_prev_state_traj[:, 1:]

        # Use the action in the environment
        action = [u0, 0]  # u0 as steering, 0 acceleration
        obs, _, done, _ = env.step(action)

        # Plot the planned trajectory in the x, y, heading - space

        Ux = trajectory_handler.get_U_x()
        dt = DELTA_T
        s = env.ego.s # get s for true dynamics

        if iteration == 0:
            if args.use_cont:
                if status == "optimal":
                    # opt_headings = np.zeros((contigency_prev_state_traj.shape[1]))
                    # for i in range(contigency_prev_state_traj.shape[1]):
                    #     _, _, opt_headings[i] = trajectory_handler.get_current_optimal_pose(opt_traj, i)
                    # # Best case
                    # proj_x, proj_y, proj_head = calculate_x_y_pos(env.ego.x, env.ego.y, env.ego.heading, opt_headings, trajectory_handler.get_U_x(), contigency_prev_state_traj[:4, :])
                    # env.world.visualizer.draw_points(np.array([proj_x, proj_y]).squeeze(axis=2), color="blue")
                    # # Worst case
                    # proj_x2, proj_y2, proj_head2 = calculate_x_y_pos(env.ego.x, env.ego.y, env.ego.heading, opt_headings, trajectory_handler.get_U_x(), contigency_prev_state_traj[4:, :])
                    # env.world.visualizer.draw_points(np.array([proj_x2, proj_y2]).squeeze(axis=2), color="orange")
                    N = contigency_prev_state_traj.shape[1]
                    # Best case
                    x_vec, y_vec, head_vec = mpc_prediction_global(opt_traj, contigency_prev_state_traj[:4,:], s, Ux, N, dt)
                    env.world.visualizer.draw_points(np.array([x_vec, y_vec]).squeeze(axis=2), color="blue")
                    # Worst case
                    x_vec_2, y_vec_2, head_vec_2 = mpc_prediction_global(opt_traj, contigency_prev_state_traj[4:,:], s, Ux, N, dt)
                    env.world.visualizer.draw_points(np.array([x_vec_2, y_vec_2]).squeeze(axis=2), color="pink")
            else:
                # opt_headings = np.zeros((prev_state_traj.shape[1]))
                # for i in range(prev_state_traj.shape[1]):
                #     _, _, opt_headings[i] = trajectory_handler.get_current_optimal_pose(opt_traj, i)
                # proj_x, proj_y, proj_head = calculate_x_y_pos(env.ego.x, env.ego.y, env.ego.heading, opt_headings, trajectory_handler.get_U_x(), prev_state_traj[:4, :])
                # env.world.visualizer.draw_points(np.array([proj_x, proj_y]).squeeze(axis=2), color="blue")
                N = prev_state_traj.shape[1]
                x_vec, y_vec, head_vec = mpc_prediction_global(opt_traj, prev_state_traj, s, Ux, N, dt)
                env.world.visualizer.draw_points(np.array([x_vec, y_vec]).squeeze(axis=2), color="pink" if args.use_ice else "blue")


        # Log
        u_vec.append(u0)
        if args.use_cont:
            state_vec.append(curr_state_c)
        else:
            state_vec.append(curr_state)
        x_y_heading_vec.append([env.ego.x, env.ego.y, env.ego.heading])  # these will be offset by one compared to state and u
        if iteration == max(prediction_horizon, 130):  # 100
            x_y_heading_vec = np.array(x_y_heading_vec).T
            env.world.visualizer.draw_points(x_y_heading_vec[:2], color="red")

            if not args.use_cont:
                plot_state_trajectories(initial_plan, "Initial plan")

                plot_state_trajectories(np.array(state_vec).T, ("Actual trajectories"))

                plot_control(u_vec)
                plt.title("Control trajectories")
                plt.show()
            else:
                plot_state_trajectories(initial_plan[:4, :], "Initial plan: Asphalt model")

                plot_state_trajectories(initial_plan[4:, :], "Initial plan: Ice model")

                plot_state_trajectories(np.array(state_vec).T[:4, :], "Actual Trajectory")

                plot_control(u_vec)
                plt.title("Control trajectories")

                plt.show()

        iteration += 1
        trajectory_handler.increment_current_index()

        if args.visualize:
            env.render()
            while time.time() - t < env.dt/2:   # Temp * 10
                pass  # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
        if done:
            env.close()
            if args.visualize:
                time.sleep(1)
