"""
state = [U_x, U_y, r, s, e, delta_psi]
U_x is the longitudinal speed
U_y is the lateral speed
r is the yaw rate
s is longitudinal distance travelled
e is the lateral error
delta_psi is the heading angle error

MPC state:
state = [U_y, r, delta_psi, e]
"""

import numpy as np
from constants import *

m = CAR_MASS
Iz = CAR_YAW_INERTIAL
a = CAR_FRONT_AXIS_DIST
b = CAR_BACK_AXIS_DIST

# linearized version of the fiala tire model
# returns force and gradient at given alpha
def linear_fiala(alpha, road_type):
    scale_factor = 1000  # kN
    if road_type == "asphalt":
        if alpha < -6:
            return 8*scale_factor, 0
        elif alpha > 6:
            return -8*scale_factor, 0
        else:
            return - 1.33 * alpha*scale_factor, -1.33
    elif road_type == "ice":
        print("found ice")
        if alpha < -1:
            return 1*scale_factor, 0
        elif alpha > 1:
            return -1*scale_factor, 0
        else:
            return - alpha*scale_factor, -1.


def get_tire_angles(U_x, U_y, r, delta):
    angle_f = np.arctan2((U_y + a*r), U_x) - np.deg2rad(delta)
    angle_r = np.arctan2((U_y - b*r), U_x)
    return np.degrees(angle_f), np.degrees(angle_r)
    # return angle_f, angle_r


def project_x_y(curr_x, curr_y, curr_heading, Ux, states):
    """
    Takes in states
    4 x N
    """
    N = states.shape[1]
    x_vec = np.zeros((N, 1))
    y_vec = np.zeros((N, 1))
    head_vec = np.zeros((N, 1))
    U_y_vec, _, _, delta_psi_vec = states[0, :], states[1, :], states[2, :], states[3, :]
    x = curr_x
    y = curr_y
    heading = curr_heading
    for i in range(N):
        U_y = U_y_vec[i]
        delta_psi = delta_psi_vec[i]
        dist_travelled = DELTA_T * np.array([Ux * np.cos(heading) - U_y * np.sin(heading),
                                            Ux * np.sin(heading) + U_y * np.cos(heading)])
        heading = np.mod(heading - delta_psi, 2*np.pi)
        x += dist_travelled[0]
        y += dist_travelled[1]

        x_vec[i] = x
        y_vec[i] = y
        head_vec[i] = heading
    return x_vec, y_vec, head_vec


def true_dynamics(state, control, Ux, kappa, road_type):
    Uy, r, e, delta_psi = state

    angle_f, angle_r = get_tire_angles(Ux, Uy, r, control)
    fy_f, _ = linear_fiala(angle_f, road_type)
    fy_r, _ = linear_fiala(angle_r, road_type)

    # dynamics equations
    U_y_dot = (fy_f + fy_r) / m - r*Ux
    r_dot = (a * fy_f - b * fy_r) / Iz
    e_dot = Uy + Ux * delta_psi
    delta_psi_dot = r - kappa*Ux
    s_dot = Ux - Uy * delta_psi

    f_true = np.array([U_y_dot, r_dot, e_dot, delta_psi_dot, s_dot])

    return f_true



def linear_dynamics(prev_values, Ux, kappa, road_type):

    # extract previous values
    Uyo, ro, _, _  = prev_values["state"]
    uo = prev_values["control"]

    ############ LINEARIZATION ################

    alpha_f_bar, alpha_r_bar = get_tire_angles(Ux, Uyo, ro, uo)

    fy_f_bar, cf = linear_fiala(alpha_f_bar, road_type)
    fy_r_bar, cr = linear_fiala(alpha_r_bar, road_type)

    A_Uy = np.array([(cf + cr)/ (m*Ux), (cf*a - cr*b)/ (m*Ux) - Ux, 0, 0 ])
    B_Uy = - cf/m
    C_Uy = (-cf*alpha_f_bar - cr*alpha_r_bar + fy_f_bar + fy_r_bar)/ m

    A_r = np.array([(a*cf - b*cr)/ (Iz*Ux), (cf* a**2 + cr* b**2)/ (Iz*Ux) - Ux, 0, 0 ])
    B_r = - (a * cf)/Iz
    C_r = (-a * cf * alpha_f_bar + b * cr * alpha_r_bar + a * fy_f_bar - b * fy_r_bar)/ Iz

    A_e = np.array([1, 0, 0, Ux])
    B_e = 0
    C_e = 0

    A_psi = np.array([0, 1, 0, 0])
    B_psi = 0
    C_psi = - kappa * Ux

    A = np.array([A_Uy, A_r, A_e, A_psi])
    B = np.array([B_Uy, B_r, B_e, B_psi])
    C = np.array([C_Uy, C_r, C_e, C_psi])

    return (A, B, C)
