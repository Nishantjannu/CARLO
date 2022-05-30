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

MAX_SLEW_RATE = 1

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
        self.slew_rate_max = MAX_SLEW_RATE
        self.traj_handler = traj_handler

    def calculate_control(self, x0, u0, prev_x_sol, prev_controls):
        """
        x0 is the initial state, shape (4, 1)
        prev_x_sol is the previous MPC solution, that we use to find the linearized dynamics matrices
            (shape 4, self.pred_horizon)
        """
        # Create optimization variables.
        u = cp.Variable((self.adim, self.pred_horizon))
        x = cp.Variable((self.sdim, self.pred_horizon+1))
        ny = cp.Variable(nonneg=True)

        QN = self.create_terminal_Q()

        constraints = [x[:,0] == x0] # First state needs to be the initial state
        cost = 0.0
        for k in range(self.pred_horizon):
            prev_vals = {
                "state": prev_x_sol[:, k],
                "control": prev_controls[:, k][0],
            }
            Ux = self.traj_handler.get_U_x()
            if k == 0:
                Q, R, v = self.create_cost_matrices(u[:,k], u0)
            else:
                Q, R, v = self.create_cost_matrices(u[:,k], u[:,k-1])
            A, B, C = linear_dynamics(prev_vals, Ux, self.traj_handler.get_kappa(k), "asphalt")


            # Add costs
            cost += cp.quad_form(x[:,k], Q)  # Add the cost x^TQx
            cost += cp.quad_form(v, R)  #v[:,k]*R*v[:,k]    # Add the cost u^TRu

            # alpha_limit = 12
            # alpha_f = (180/np.pi)*(x[0,k] + a*x[1,k])/Ux - u[0,k]
            # alpha_r = (180/np.pi)*(x[0,k] - b*x[1,k])/Ux
            # cost += (1/36)*cp.square(alpha_f)
            # cost += (1/36)*cp.square(alpha_r)
            # constraints += [alpha_f <= alpha_limit, alpha_r <= alpha_limit]
            constraints += [x[:,k+1] == x[:, k] + self.delta_t*A@x[:,k] + self.delta_t*B*u[:,k] + self.delta_t*C]             # Add the system dynamics x(k+1) = A*x(k) + B*u(k) + C
            constraints += [-self.steering_max <= u[:,k], u[:,k] <= self.steering_max] # Constraints for the control signal
            constraints += [-self.slew_rate_max <= v, v <= self.slew_rate_max] # Constraints for the control signal

            # Within bounds constraint
            # Better to put one ny for each loop iteration? Could do that too
            constraints += [cp.abs(x[2, k]) - self.traj_handler.get_lane_bounds(k) <= ny]  # nr will get e

        cost += 10000 * ny

        cost += cp.quad_form(x[:, self.pred_horizon], QN)
        # alpha_limit = 25
        # alpha_f = (180/np.pi)*(x[0,self.pred_horizon] + a*x[1,self.pred_horizon])/Ux - u[0,self.pred_horizon-1]
        # alpha_r = (180/np.pi)*(x[0,self.pred_horizon] - b*x[1,self.pred_horizon])/Ux
        # cost += (1/36)*cp.abs(alpha_f) + (1/36)*cp.abs(alpha_r)
        # constraints += [cp.abs(alpha_f) <= alpha_limit, cp.abs(alpha_r) <= alpha_limit]

        # Form and solve problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        sol = prob.solve(solver=cp.ECOS)
        print("cost is: ", prob.value)
        print("ny is: ", ny.value)
        status = prob.status
        if status != "optimal":
            print("\n\n\nStatus NOT OPTIMAL (status is):", status, "\n\n\n")

        return u.value, x.value



    def create_terminal_Q(self):
        Q = np.zeros((self.sdim, self.sdim))
        Q[2, 2] = 1
        Q[3, 3] = 1
        return Q

    def create_cost_matrices(self, u_k, u_k_prev):
        Q = np.zeros((self.sdim, self.sdim))
        # Q[0, 0] = 1
        # Q[1, 1] = 1
        Q[2, 2] = 1
        Q[3, 3] = 1

        R = np.zeros((self.adim, self.adim))
        R[self.adim-1, self.adim-1] = self.R_val

        v = u_k - u_k_prev

        return Q, R, v



class MPC_ice:
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
        self.slew_rate_max = MAX_SLEW_RATE
        self.traj_handler = traj_handler

    def calculate_control(self, x0, u0, prev_x_sol, prev_controls):
        """
        x0 is the initial state, shape (4, 1)
        prev_x_sol is the previous MPC solution, that we use to find the linearized dynamics matrices
            (shape 4, self.pred_horizon)
        """
        # Create optimization variables.
        u = cp.Variable((self.adim, self.pred_horizon))
        x = cp.Variable((self.sdim, self.pred_horizon+1))
        ny = cp.Variable(nonneg=True)

        QN = self.create_terminal_Q()

        constraints = [x[:,0] == x0] # First state needs to be the initial state
        cost = 0.0
        for k in range(self.pred_horizon):
            prev_vals = {
                "state": prev_x_sol[:, k],
                "control": prev_controls[:, k][0],
            }
            Ux = self.traj_handler.get_U_x()
            if k == 0:
                Q, R, v = self.create_cost_matrices(u[:,k], u0)
            else:
                Q, R, v = self.create_cost_matrices(u[:,k], u[:,k-1])
            A, B, C = linear_dynamics(prev_vals, Ux, self.traj_handler.get_kappa(k), "ice")


            # Add costs
            # cost += cp.quad_form(x[:,k], Q)  # Add the cost x^TQx
            # cost += cp.quad_form(v, R)  #v[:,k]*R*v[:,k]    # Add the cost u^TRu

            # alpha_limit = 12
            # alpha_f = (180/np.pi)*(x[0,k] + a*x[1,k])/Ux - u[0,k]
            # alpha_r = (180/np.pi)*(x[0,k] - b*x[1,k])/Ux
            # cost += (1/36)*cp.square(alpha_f)
            # cost += (1/36)*cp.square(alpha_r)
            # constraints += [alpha_f <= alpha_limit, alpha_r <= alpha_limit]
            constraints += [x[:,k+1] == x[:, k] + self.delta_t*A@x[:,k] + self.delta_t*B*u[:,k] + self.delta_t*C]             # Add the system dynamics x(k+1) = A*x(k) + B*u(k) + C
            constraints += [-self.steering_max <= u[:,k], u[:,k] <= self.steering_max] # Constraints for the control signal
            constraints += [-self.slew_rate_max <= v, v <= self.slew_rate_max] # Constraints for the control signal

            # Within bounds constraint
            # Better to put one ny for each loop iteration? Could do that too
            constraints += [cp.abs(x[2, k]) - self.traj_handler.get_lane_bounds(k) <= ny]  # nr will get e

        cost += 10000 * ny

        cost += cp.quad_form(x[:, self.pred_horizon], QN)
        # alpha_limit = 25
        # alpha_f = (180/np.pi)*(x[0,self.pred_horizon] + a*x[1,self.pred_horizon])/Ux - u[0,self.pred_horizon-1]
        # alpha_r = (180/np.pi)*(x[0,self.pred_horizon] - b*x[1,self.pred_horizon])/Ux
        # cost += (1/36)*cp.abs(alpha_f) + (1/36)*cp.abs(alpha_r)
        # constraints += [cp.abs(alpha_f) <= alpha_limit, cp.abs(alpha_r) <= alpha_limit]

        # Form and solve problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        sol = prob.solve(solver=cp.ECOS)
        print("cost is: ", prob.value)
        print("ny is: ", ny.value)
        status = prob.status
        if status != "optimal":
            print("\n\n\nStatus NOT OPTIMAL (status is):", status, "\n\n\n")

        return u.value, x.value



    def create_terminal_Q(self):
        Q = np.zeros((self.sdim, self.sdim))
        Q[2, 2] = 1
        Q[3, 3] = 1
        return Q

    def create_cost_matrices(self, u_k, u_k_prev):
        Q = np.zeros((self.sdim, self.sdim))
        # Q[0, 0] = 1
        # Q[1, 1] = 1
        Q[2, 2] = 1
        Q[3, 3] = 1

        R = np.zeros((self.adim, self.adim))
        R[self.adim-1, self.adim-1] = self.R_val

        v = u_k - u_k_prev

        return Q, R, v





class Contigency_MPC:
    """
    Contigency MPC class
    - inherit from common MPC class? Or write some duplicate code. Would be good with some inheritance

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
        self.slew_rate_max = MAX_SLEW_RATE
        self.traj_handler = traj_handler

    def calculate_control(self, x0, u0, prev_x_sol, prev_controls):
        """
        x0 is the initial state, shape (8, 1)
        prev_x_sol is the previous MPC solution, that we use to find the linearized dynamics matrices
            (shape 8, self.pred_horizon)
        u = u0 should have shape [2, 1]
        """
        # Create optimization variables.
        u = cp.Variable((self.adim, self.pred_horizon))
        x = cp.Variable((self.sdim, self.pred_horizon+1))
        ny = cp.Variable(nonneg=True)

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
            Ux = self.traj_handler.get_U_x()
            if k == 0:
                Q, R, v = self.create_cost_matrices(u[:,k], u0)
            else:
                Q, R, v = self.create_cost_matrices(u[:,k], u[:,k-1])
            A, B, C = contigency_linear_dynamics(prev_vals_nom, prev_vals_c, Ux, self.traj_handler.get_kappa(k), "asphalt", "ice")

            # Add costs
            cost += cp.quad_form(x[:,k], Q)  # Add the cost x^TQx
            cost += cp.quad_form(v, R)    # Add the cost u^TRu

            # Add constraints
            # print("shape of x:", x[:, k].shape, "shape of A", A.shape, "shape of B", B.shape, "shape of u", u[:,k].shape, "shape of C", C.shape)

            constraints += [x[:,k+1] == x[:, k] + self.delta_t*A@x[:,k] + self.delta_t*B@u[:,k] + self.delta_t*C]             # Add the system dynamics x(k+1) = A*x(k) + B*u(k) + C
            constraints += [-self.steering_max <= u[:,k], u[:,k] <= self.steering_max] # Constraints for the control signal
            constraints += [-self.slew_rate_max <= v, v <= self.slew_rate_max] # Constraints for the control signal

            # Within bounds constraint
            # Better to put one ny for each loop iteration? Could do that too
            constraints += [cp.abs(x[6, k]) <= self.traj_handler.get_lane_bounds(k) + ny]  # nr will get ice-controller-e

        cost += 10000 * ny

        cost += cp.quad_form(x[:, self.pred_horizon], QN)

        # Form and solve problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        sol = prob.solve(solver=cp.ECOS)
        status = prob.status
        print("Cost is: ", prob.value, "ny is:", ny.value)
        if status != "optimal" or prob.value > 1000:
            print("\n\n\nStatus NOT OPTIMAL (status is):", status, "\n\n\n")
            status = "sub_optimal"
            # u_new = np.zeros_like(u.value)
            # u_new[:, 0] = u0
            # return u_new, x.value, status

        # Final alphas
        # alpha_limit = 25
        # alpha_f_act = (180/np.pi)*(x.value[0,self.pred_horizon] + a*x.value[1,self.pred_horizon])/Ux - u.value[0,self.pred_horizon-1]
        # alpha_f_act = (180/np.pi)*(x.value[0,self.pred_horizon] - b*x.value[1,self.pred_horizon])/Ux
        # if np.abs(alpha_f_act) > alpha_limit or np.abs(alpha_f_act) > alpha_limit:
        #     print("final alphas above limit of:", alpha_limit, "selecting nominal trajectory to linearize around")
        #     u_new = np.zeros_like(u.value)
        #     u_new[:, 0] = u.value[:, 0]
        #     return u_new, np.zeros_like(x.value)

        return u.value, x.value, status



    def create_terminal_Q(self):
        Q = np.zeros((self.sdim, self.sdim))
        Q[self.sdim-2, self.sdim-2] = 1
        Q[self.sdim-1, self.sdim-1] = 1
        return Q

    def create_cost_matrices(self, u_k, u_k_prev):
        Q = np.zeros((self.sdim, self.sdim))
        Q[self.sdim//2-2, self.sdim//2-2] = 1  # division by 2 sets for the nominal states
        Q[self.sdim//2-1, self.sdim//2-1] = 1

        R = np.eye(self.adim)
        R[self.adim//2-1, self.adim//2-1] = self.R_val

        v = u_k - u_k_prev  # 2 x 1 vector

        return Q, R, v
