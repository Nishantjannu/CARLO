#!/usr/bin/env python3
import numpy as np
import gym_carlo
import gym
import time
import argparse
from gym_carlo.envs.interactive_controllers import KeyboardController
from scipy.stats import multivariate_normal
from train_ildist import NN
from utils import *
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="circularroad, lanechange", default="lanechange")
    args = parser.parse_args()
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, '--scenario argument is invalid!'
    assert scenario_name != 'intersection', '--scenario cannot be intersection for shared_autonomy.py' # we don't have the optimal policy for that
    
    env = gym.make(scenario_name + 'Scenario-v0', goal=len(goals[scenario_name]))
    
    nn_models = {}
    for goal in goals[scenario_name]:
        nn_models[goal] = NN(obs_sizes[scenario_name],2)
        nn_models[goal].load_weights('./policies/' + scenario_name + '_' + goal + '_ILDIST')
        
    max_steering = 0.05
    max_throttle = 0.

    env.T = 200*env.dt - env.dt/2. # Run for at most 200dt = 20 seconds
    for _ in range(10):
        env.seed(int(np.random.rand()*1e6))
        obs, done = env.reset(), False
        env.render()
        
        optimal_action = {}
        interactive_policy = KeyboardController(env.world, steering_lims[scenario_name])
        scores = np.array([[1./len(goals[scenario_name])]*len(goals[scenario_name])]*100)
        while not done:
            t = time.time()
            obs = np.array(obs).reshape(1,-1)
            
            for goal_id in range(len(goals[scenario_name])):
                if args.scenario.lower() == 'circularroad':
                    optimal_action[goals[scenario_name][goal_id]] = optimal_act_circularroad(env, goal_id)
                elif args.scenario.lower() == 'lanechange':
                    optimal_action[goals[scenario_name][goal_id]] = optimal_act_lanechange(env, goal_id)
            
            ######### Your code starts here #########
            # We want to compute the expected human action and generate the robot action.
            # The following variables should be sufficient:
            # - scenario_name keeps the name of the scenario, e.g. 'lanechange'
            # - goals[scenario_name] is the list of goals G, e.g. ['left', 'right'].
            # - nn_models[goal] is the trained mixed density network, e.g. nn_models['left']
            # - scores (100 x |G| numpy array) keeps the predicted probabilities of goals for last 100 steps (from earlier to later)
            # - obs (1 x dim(O) numpy array) is the current observation
            # - optimal_action[goal] gives the optimal action (1x2 numpy array) for the specified goal, e.g. optimal_action['left']
            # - max_steering and max_throttle are the constraints, i.e. np.abs(a_robot[0]) <= max_steering and np.abs(a_robot[1]) <= max_throttle must be satisfied.
            # At the end, your code should set a_robot variable as a 1x2 numpy array that consists of steering and throttle values, respectively
            # HINT: You can use np.clip to threshold a_robot with respect to the magnitude constraints

            exp_a_human = np.zeros((1, 2))
            a_robot = np.zeros((1, 2))
            for goal_nr, goal in enumerate(goals[scenario_name]):
                m_g = optimal_action[goal]
                P_g = np.mean(scores[:, goal_nr])
                model_g = nn_models[goal]

                output = np.array(model_g.call(obs))[0]
                mean = output[0:2].reshape((1, 2))  # What we are seeking is the definition of the mean, so we use the mean
                exp_a_human += P_g * mean
                a_robot += P_g * m_g - P_g * mean

            a_robot[0, 0] = np.clip(a_robot[0, 0], -max_steering, max_steering)
            a_robot[0, 1] = np.clip(a_robot[0, 1], -max_throttle, max_throttle)

            ########## Your code ends here ##########
            
            a_human = np.array([interactive_policy.steering, optimal_action[goals[scenario_name][0]][0,1]]).reshape(1,-1)
            
            ######### Your code starts here #########
            # Having seen the human_action, we want to infer the human intent.
            # The following variables should be sufficient:
            # - scenario_name keeps the name of the scenario, e.g. 'lanechange'
            # - goals[scenario_name] is the list of goals G, e.g. ['left', 'right']
            # - nn_models[goal] is the trained mixed density network, e.g. nn_models['left']
            # - obs (1 x dim(O) numpy array) is the current observation
            # - a_human (1 x 2 numpy array) is the current action the user took when the observation is obs
            # At the end, your code should set probs variable as a 1 x |G| numpy array that consists of the probability of each goal under obs and a_human
            # HINT: This should be very similar to the part in intent_inference.py 

            denonminator = 0
            probs = np.zeros((1, len(goals[scenario_name])))
            for goal_nr, goal in enumerate(goals[scenario_name]):
                mod = nn_models[goal]
                output = np.array(mod.call(obs))[0]
                mean = output[0:2]
                cov_mat = output[2:6].reshape(2, 2)
                cov = np.multiply(cov_mat, cov_mat.T)
                try:
                    prob = multivariate_normal.pdf(a_human, mean=mean, cov=cov + np.eye(2, 2) * 1e-4)
                except:
                    prob = multivariate_normal.pdf(a_human, mean=mean, cov=cov + np.eye(2, 2) * 1e-1)
                probs[0, goal_nr] = prob
                denonminator += prob

            for prob_nr in range(len(probs)):
                probs[0, prob_nr] = probs[0, prob_nr] / denonminator

            ########## Your code ends here ##########

            # shift the scores and append the latest one
            scores[:-1] = scores[1:]
            scores[-1] = probs
            
            action = a_robot + a_human
            obs,_,done,_ = env.step(action.reshape(-1))
            env.render()
            while time.time() - t < env.dt/2.: pass # runs 2x speed. This is not great, but time.sleep() causes problems with the interactive controller
            if done:
                env.close()
                time.sleep(1)
