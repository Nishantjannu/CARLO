"""
Vanilla Model Predictive Control

Author: Albin Mosskull
04-20-2022
"""

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from dynamics import linear_dynamics
from nominal_trajectory import Nominal_Trajectory_Handler
from constants import DELTA_T


class MPC:
    """
    Vanilla MPC class

    # State #
    x = [U_y, r, delta_psi, e]
    U_y is the lateral speed
    r is the yaw rate
    delta_psi is the heading angle error
    e is the lateral error from the path
    """
    def __init__(self, pred_horizon, traj_handler):
        self.pred_horizon = pred_horizon
        self.sdim = 4  # We have 4 state variables
        self.adim = 1
        self.R_val = 0.01
        self.delta_t = DELTA_T
        self.W_val = 50
        self.steering_max = np.pi / 2  # Todo update
        self.slew_rate_max = 20  # Todo update
        self.traj_handler = traj_handler

    def calculate_control(self, x0, prev_x_sol, prev_controls):
        """
        x0 is the initial state, shape (4, 1)
        prev_x_sol is the previous MPC solution, that we use to find the linearized dynamics matrices
            (shape 4, self.pred_horizon)
        """
        # Create optimization variables.
        u = cp.Variable((self.adim, self.pred_horizon))
        x = cp.Variable((self.sdim, self.pred_horizon+1))

        QN = self.create_terminal_Q()

        constraints = [x[:,0] == x0] # First constraints needs to be the initial state
        cost = 0.0
        for k in range(self.pred_horizon):
            prev_vals = {
                "state": prev_x_sol[:, k],
                "control": prev_controls[:, k],
            }
            A, B, C = linear_dynamics(prev_vals, self.traj_handler.get_U_x(), self.traj_handler.get_kappa(k), "asphalt")
            Q, R, v, W = self.create_cost_matrices(u[:,k], u[:,k-1])

            # Add costs
            cost += cp.quad_form(x[:,k], Q)  # Add the cost x^TQx
            # cost += u[:,k]*R*u[:,k] # cp.quad_form(u[k], R)    # Add the cost u^TRu

            # Add constraints
            # print("shape of x:", x[:, k].shape, "shape of A", A.shape, "shape of B", B.shape, "shape of u", u[:,k].shape, "shape of C", C.shape)

            # TODO multiply by delta t here
            constraints += [x[:,k+1] == x[:, k] + self.delta_t*A@x[:,k] + self.delta_t*B*u[:,k] + self.delta_t*C]             # Add the system dynamics x(k+1) = A*x(k) + B*u(k) + C
            # constraints += [-self.steering_max <= u[:,k], u[:,k] <= self.steering_max] # Constraints for the control signal
            # constraints += [-self.slew_rate_max <= v, v <= self.steering_max] # Constraints for the control signal
            # TODO add constraint depending on the friction environment

        cost += cp.quad_form(x[:, self.pred_horizon], QN)

        # Form and solve problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        sol = prob.solve(solver=cp.ECOS)
        # warnings.filterwarnings("ignore")
        return u.value, x.value



    def create_terminal_Q(self):
        Q = np.zeros((self.sdim, self.sdim))
        Q[2, 2] = 1
        Q[3, 3] = 1
        return Q

    def create_cost_matrices(self, u_k, u_k_prev):
        Q = np.zeros((self.sdim, self.sdim))
        Q[2, 2] = 1
        Q[3, 3] = 1

        R = self.R_val

        v = u_k - u_k_prev

        W = self.W_val
        return Q, R, v, W
