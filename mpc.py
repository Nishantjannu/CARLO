"""
General Model Predictive Control

Author: Albin Mosskull
04-20-2022
"""

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp



class MPC:
    """
    Vanilla MPC class

    # State #
    x = [x_nom, x_c].T
    x_nom = x_c = [U_y, r, delta_psi, e]
    U_y is the lateral speed
    r is the yaw rate
    delta_psi is the heading angle error
    e is the lateral error from the path
    """
    def __init__(self, pred_horizon=20):
        self.pred_horizon = pred_horizon
        self.sdim = 4  # We have 4 state variables
        self.adim = 1
        self.R_val = 0.01
        self.W = 50
        self.steering_max = np.pi / 2  # Todo update
        self.slew_rate_max = 20  # Todo update


    def calculate_control(x0):
        """
        x0 is the initial state, shape (4, 1)
        """
        # Create optimization variables.
        u = cp.Variable((self.adim, PREDICTION_HORIZON))
        x = cp.Variable((self.sdim, PREDICTION_HORIZON+1))

        QN = create_terminal_Q()
        x_ref = np.zeros((sdim, 1))

        constraints = [x[:,0] == x0] # First constraints needs to be the initial state
        cost = 0.0
        for k in range(PREDICTION_HORIZON):
            # Retrieve the dynamics and cost matrices for the current iteration
            Ad, Bd, Dd = create_matrices_linear(x[:,k])  # x[:, k] is the past state
            Q, R, v, W = create_cost_matrices(u[k], u[k-1])

            # Add costs
            cost += cp.quad_form(x[:,k] - xref, Q) + cp.quad_form(u[k], R)    # Add the cost function sum(x^TQx + u^TRu)

            # Add constraints
            constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[k] + Dd]             # Add the system dynamics x(k+1) = Ad*x(k) + Bd*u(k) + Dd
            constraints += [-self.steering_max <= u[k], u[k] <= self.steering_max] # Constraints for the control signal
            constraints += [-self.slew_rate_max <= v[k], v[k] <= self.steering_max] # Constraints for the control signal

        cost += cp.quad_form(x[:, PREDICTION_HORIZON] - xref, QN)

        # Form and solve problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        sol = prob.solve(solver=cp.ECOS)
        # warnings.filterwarnings("ignore")
        return u[0].value, x.value


    def create_linear_dynamic_matrices(self, x):
        raise NotImplementedError
        # return A, B, D


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
