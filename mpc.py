"""
Vanilla Model Predictive Control

Author: Albin Mosskull
04-20-2022
"""

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from dynamics import linear_dynamics, contigency_linear_dynamics
from nominal_trajectory import Nominal_Trajectory_Handler
from constants import DELTA_T, CAR_FRONT_AXIS_DIST as a, CAR_BACK_AXIS_DIST as b

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
        self.steering_max = 5
        self.slew_rate_max = 0.5
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

        constraints = [x[:,0] == x0] # First state needs to be the initial state
        cost = 0.0
        for k in range(self.pred_horizon):
            prev_vals = {
                "state": prev_x_sol[:, k],
                "control": prev_controls[:, k],
            }
            Ux = self.traj_handler.get_U_x()
            A, B, C = linear_dynamics(prev_vals, Ux, self.traj_handler.get_kappa(k), "asphalt")
            Q, R, v = self.create_cost_matrices(u[:,k], u[:,k-1])

            # Add costs
            cost += cp.quad_form(x[:,k], Q)  # Add the cost x^TQx
            # cost += cp.quad_form(v, R)  #v[:,k]*R*v[:,k]    # Add the cost u^TRu

            # alpha_f = (180/np.pi)*(x[0,k] + a*x[1,k])/Ux - u[0,k]
            # alpha_r = (180/np.pi)*(x[0,k] - b*x[1,k])/Ux
            # constraints += [cp.abs(alpha_f) <= 6, cp.abs(alpha_r) <= 6]
            constraints += [x[:,k+1] == x[:, k] + self.delta_t*A@x[:,k] + self.delta_t*B*u[:,k] + self.delta_t*C]             # Add the system dynamics x(k+1) = A*x(k) + B*u(k) + C
            constraints += [-self.steering_max <= u[:,k], u[:,k] <= self.steering_max] # Constraints for the control signal
            constraints += [-self.slew_rate_max <= v, v <= self.steering_max] # Constraints for the control signal
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

        R = np.zeros((self.adim, self.adim))
        R[self.adim-1, self.adim-1] = self.R_val

        v = u_k - u_k_prev

        return Q, R, v


class Contigency_MPC:
    """
    Contigency MPC class
    - inherit from common MPC class? Or write some duplicate code

    # State #
    [x_nom, x_c]
    x_nom = x_c = [U_y, r, delta_psi, e]
    U_y is the lateral speed
    r is the yaw rate
    delta_psi is the heading angle error
    e is the lateral error from the path

    Crucial compared to the nominal MPC is the duplicate state space, and that the first actions are enforced to be the same
    """
    def __init__(self, pred_horizon, traj_handler):
        self.pred_horizon = pred_horizon
        self.sdim = 8
        self.adim = 2
        self.R_val = 0.01
        self.delta_t = DELTA_T
        self.steering_max = 5
        self.slew_rate_max = 0.1
        self.traj_handler = traj_handler

    def calculate_control(self, x0, prev_x_sol, prev_controls):
        """
        x0 is the initial state, shape (8, 1)
        prev_x_sol is the previous MPC solution, that we use to find the linearized dynamics matrices
            (shape 8, self.pred_horizon)
        """
        # Create optimization variables.
        u = cp.Variable((self.adim, self.pred_horizon))
        x = cp.Variable((self.sdim, self.pred_horizon+1))

        QN = self.create_terminal_Q()

        constraints = [x[:,0] == x0] # First state needs to be the initial state
        constraints += [u[0, 0] == u[1, 0]]  # Enforce first state identical for both dynamics
        cost = 0.0
        for k in range(self.pred_horizon):
            prev_vals_nom = {
                "state": prev_x_sol[:4, k],
                "control": prev_controls[0, k],
            }
            prev_vals_c = {
                "state": prev_x_sol[4:, k],
                "control": prev_controls[1, k],
            }
            A, B, C = contigency_linear_dynamics(prev_vals_nom, prev_vals_c, self.traj_handler.get_U_x(), self.traj_handler.get_kappa(k), "asphalt", "ice")
            Q, R, v = self.create_cost_matrices(u[:,k], u[:,k-1])

            # Add costs
            cost += cp.quad_form(x[:,k], Q)  # Add the cost x^TQx, Q will be correct shape for 8 state setup
            # cost += v[:,k]*R*v[:,k] # cp.quad_form(v[k], R)    # Add the cost u^TRu

            # Add constraints
            print("shape of x:", x[:, k].shape, "shape of A", A.shape, "shape of B", B.shape, "shape of u", u[:,k].shape, "shape of C", C.shape)

            constraints += [x[:,k+1] == x[:, k] + self.delta_t*A@x[:,k] + self.delta_t*B@u[:,k] + self.delta_t*C]             # Add the system dynamics x(k+1) = A*x(k) + B*u(k) + C
            # constraints += [-self.steering_max <= u[:,k], u[:,k] <= self.steering_max] # Constraints for the control signal
            # constraints += [-self.slew_rate_max <= v, v <= self.steering_max] # Constraints for the control signal

        cost += cp.quad_form(x[:, self.pred_horizon], QN)

        # Form and solve problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        sol = prob.solve(solver=cp.ECOS)
        # warnings.filterwarnings("ignore")
        return u.value, x.value



    def create_terminal_Q(self):
        Q = np.zeros((self.sdim, self.sdim))
        Q[self.sdim-2, self.sdim-2] = 1
        Q[self.sdim-1, self.sdim-1] = 1
        return Q

    def create_cost_matrices(self, u_k, u_k_prev):
        Q = np.zeros((self.sdim, self.sdim))
        Q[self.sdim-2, self.sdim-2] = 1
        Q[self.sdim-1, self.sdim-1] = 1

        R = np.eye(self.adim)
        R[self.adim-1, self.adim-1]
        R = self.R_val

        v = u_k - u_k_prev  # 2 x 1 vector

        return Q, R, v
