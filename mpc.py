"""
General Model Predictive Control

Author: Albin Mosskull
04-20-2022
"""


import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp


PREDICTION_HORIZON = 30


class MPC:
    """
    Contingency MPC class

    # State #
    x = [x_nom, x_c].T
    x_nom = x_c = [U_y, r, delta_psi, e]
    U_y is the lateral speed
    r is the yaw rate
    delta_psi is the heading angle error
    e is the lateral error from the path
    """
    def __init__(pred_horizon, u_max):
        self.pred_horizon = pred_horizon
        self.xdim = 4  # We have 4 state variables
        self.R_nom_val = 0.01
        self.W11 = 50
        self.W22 = 500


    def calculate_control():
        """

        """
        # Even if air drag has been choosen the first mpc needs to calculate with A and B matrices for the ideal case
        Ad, Bd, Dd = self.create_cost_matrices()  # Create A, B and D matrices for the ideal case
        Q, R = create_cost_matrices()
        QN = Q

        x0 = []

        # Create two scalar optimization variables.
        u = cp.Variable((NUM_CARS-1, PREDICTION_HORIZON))
        x = cp.Variable((2*NUM_CARS-1, PREDICTION_HORIZON+1))

        constraints = [x[:,0] == x0] # First constraints needs to be the initial state
        cost = 0.0 # Define the cost function
        for k in range(PREDICTION_HORIZON):
            Ad, Bd, Dd = create_matrices_linear(x_last[:,k+1])
            cost += cp.quad_form(x[:,k] - xref, Q) + cp.quad_form(u[:,k], R)    # Add the cost function sum(x^TQx + u^TRu)
            constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k] + Dd]             # Add the system x(k+1) = Ad*x(k) + Bd*u(k) + Dd
            constraints += [MIN_ACCELERATION <= u[:,k], u[:,k] <= MAX_ACCELERATION] # Constraints for the control signal
        cost += cp.quad_form(x[:,PREDICTION_HORIZON] - xref, QN)

        # Form and solve problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        sol = prob.solve(solver=cp.ECOS)
        warnings.filterwarnings("ignore")
        return u[:,0].value, x.value


    def create_linear_dynamic_matrices(self, x):
        raise NotImplementedError
        # return A, B, D


    def create_cost_matrices(self, u):
        Q_nom = np.zeros((self.xdim, self.xdim))
        Q = np.zeros((self.xdim*self.xdim, self.xdim*self.xdim))
        Q[12:16, 12:16] = Q_nom

        R_nom = self.R_nom_val
        R = np.zeros((2, 2))
        R[0, 0] = R_nom

        v = np.zeros((2, 1))

        W = np.zeros((2, 2))
        W[0, 0] = self.W11
        W[1, 1] = self.W22
        return Q, R, v, W
