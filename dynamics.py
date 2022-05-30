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
import matplotlib.pyplot as plt
from constants import *

# Assign values to the car's parameters based on the constants
m = CAR_MASS
Iz = CAR_YAW_INERTIAL
a = CAR_FRONT_AXIS_DIST
b = CAR_BACK_AXIS_DIST


def linear_fiala(alpha, road_type):
    """
    linearized version of the fiala tire model
    returns force and gradient at given alpha
    """
    scale_factor = 1000  # kN
    saturation_derivative = 0.00  # 0
    if road_type == "asphalt":
        if alpha < -6:
            return 8*scale_factor, saturation_derivative*scale_factor
        elif alpha > 6:
            return -8*scale_factor, saturation_derivative*scale_factor
        else:
            return - 1.33 * alpha*scale_factor, -1.33*scale_factor
    # elif road_type == "ice":
    #     if alpha < -1:
    #         return 1*scale_factor, saturation_derivative*scale_factor
    #     elif alpha > 1:
    #         return -1*scale_factor, saturation_derivative*scale_factor
    #     else:
    #         return - alpha*scale_factor, -1.*scale_factor
    elif road_type == "ice":
        if alpha < -1:
            return 1*scale_factor, saturation_derivative*scale_factor
        elif alpha > 1:
            return -1*scale_factor, saturation_derivative*scale_factor
        else:
            return -1*alpha*scale_factor, -1.*scale_factor



def get_s_nominal(opt_traj):
    x_poses = opt_traj[:,0]
    y_poses = opt_traj[:,1]
    angles = opt_traj[:,2]

    sx = np.square(x_poses[:-1] - x_poses[1:])
    sy = np.square(y_poses[:-1] - y_poses[1:])
    s_nom = np.sqrt(sx + sy)

    for i in range(1,s_nom.shape[0]):
        s_nom[i] += s_nom[i-1]

    return s_nom

def find_closest_nominal(opt_traj, s_nom, s):

    res = next(x for x, val in enumerate(s_nom) if val >= s)

    x_nom = opt_traj[:,0][res]
    y_nom = opt_traj[:,1][res]
    theta = opt_traj[:,2][res]

    return x_nom, y_nom, theta

def mpc_prediction_global(opt_traj, states, s, Ux, N, dt):
    x_vec = np.zeros((N, 1))
    y_vec = np.zeros((N, 1))
    head_vec = np.zeros((N, 1))

    s_nom = get_s_nominal(opt_traj)

    for i in range(N):
        U_y, _, e, delta_psi = states[:, i]

        s += (Ux*np.cos(delta_psi) - U_y * np.sin(delta_psi)) * dt

        x_nom, y_nom, theta = find_closest_nominal(opt_traj, s_nom, s)

        x_vec[i] = x_nom - e * np.sin(theta)
        y_vec[i] = y_nom + e * np.cos(theta)
        head_vec[i] = theta + delta_psi

    return x_vec, y_vec, head_vec


def get_tire_angles(U_x, U_y, r, delta):
    """
    Assumes delta in degrees
    """
    angle_f = np.arctan2((U_y + a*r), U_x) - np.deg2rad(delta)
    angle_r = np.arctan2((U_y - b*r), U_x)
    return np.degrees(angle_f), np.degrees(angle_r)


def linear_get_tire_angles(U_x, U_y, r, delta):
    """
    Assumes delta in degrees
    """
    angle_f = ((U_y + a*r)/ U_x) - np.deg2rad(delta)
    angle_r = ((U_y - b*r)/ U_x)
    return np.degrees(angle_f), np.degrees(angle_r)


def calculate_x_y_pos(curr_x, curr_y, curr_heading, opt_headings, Ux, states):
    """
    Takes in states (4 x N):
    [U_y
    r
    e
    delta_psi]
    delta_psis need to be shifted one value into the future due to how x and y are updated

    N is automatically assumed as the horizon for how far to project the x and y.

    Projects curr_x and curr_y into the future.
    """
    # Create vectors for storing the results
    N = states.shape[1]
    x_vec = np.zeros((N, 1))
    y_vec = np.zeros((N, 1))
    head_vec = np.zeros((N, 1))
    U_y_vec, _, _, delta_psi_vec = states[0, :], states[1, :], states[2, :], states[3, :]

    # Initialize the position projection loop
    x = curr_x
    y = curr_y
    heading = curr_heading
    for i in range(N):
        U_y = U_y_vec[i]
        delta_psi = delta_psi_vec[i]
        opt_heading = opt_headings[i]
        dist_travelled = DELTA_T * np.array([(Ux) * np.cos(heading) - (U_y) * np.sin(heading),  # Ux is constant
                                            (Ux) * np.sin(heading) + (U_y) * np.cos(heading)])

        heading = np.mod(opt_heading + delta_psi, 2*np.pi)  # calculate dist_travelled before updating heading, put + instead of minus for right hand turn
        x += dist_travelled[0]
        y += dist_travelled[1]

        # Store the values
        x_vec[i] = x
        y_vec[i] = y
        head_vec[i] = heading
    return x_vec, y_vec, head_vec


def true_dynamics(state, control, Ux, kappa, road_type):
    Uy, r, e, delta_psi = state

    angle_f, angle_r = get_tire_angles(Ux, Uy, r, control)
    fy_f, _ = linear_fiala(angle_f, road_type)
    fy_r, _ = linear_fiala(angle_r, road_type)
    print("True dynamics: fy_f, fy_r:", fy_f, fy_r)

    # dynamics equations
    U_y_dot = (fy_f + fy_r) / m - r*Ux
    r_dot = (a * fy_f - b * fy_r) / Iz
    e_dot = Uy + Ux * delta_psi
    delta_psi_dot = r - kappa*Ux
    s_dot = Ux - Uy * delta_psi
    # e_dot = Uy*np.cos(delta_psi) + Ux * np.sin(delta_psi)
    # s_dot = Ux*np.cos(delta_psi) - Uy * np.sin(delta_psi)

    f_true = np.array([U_y_dot, r_dot, e_dot, delta_psi_dot, s_dot])

    return f_true


def contigency_linear_dynamics(prev_vals_nom, prev_vals_c, Ux, kappa, road_type_nom, road_type_c):
    sdim = len(prev_vals_nom["state"])
    adim = 1  # should be done in a better way...

    A_tot = np.zeros((2*sdim, 2*sdim))
    B_tot = np.zeros((2*sdim, 2*adim))
    C_tot = np.zeros((2*sdim,))

    A_nom, B_nom, C_nom = linear_dynamics(prev_vals_nom, Ux, kappa, road_type_nom)
    A_c, B_c, C_c = linear_dynamics(prev_vals_c, Ux, kappa, road_type_c)

    A_tot[:sdim, :sdim] = A_nom
    A_tot[sdim:, sdim:] = A_c
    B_tot[:sdim, 0] = B_nom
    B_tot[sdim:, 1] = B_c
    C_tot[:sdim] = C_nom
    C_tot[sdim:] = C_c

    return(A_tot, B_tot, C_tot)


def linear_dynamics(prev_values, Ux, kappa, road_type):

    # extract previous values
    Uyo, ro, _, _  = prev_values["state"]
    uo = prev_values["control"]


    ############ LINEARIZATION ################

    alpha_f_bar, alpha_r_bar = linear_get_tire_angles(Ux, Uyo, ro, uo)

    fy_f_bar, cf = linear_fiala(alpha_f_bar, road_type)
    fy_r_bar, cr = linear_fiala(alpha_r_bar, road_type)
    # print("Linear dynamics: fy_f, fy_r:", fy_f_bar, fy_r_bar)

    A_Uy = np.array([(cf + cr)/ (m*Ux), (cf*a - cr*b)/ (m*Ux) - Ux, 0, 0 ])
    B_Uy = - cf/m
    C_Uy = (-cf*alpha_f_bar - cr*alpha_r_bar + fy_f_bar + fy_r_bar)/ m

    A_r = np.array([(a*cf - b*cr)/ (Iz*Ux), (cf* a**2 + cr* b**2)/ (Iz*Ux), 0, 0 ])
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


if __name__ == "__main__":
    x = np.arange(-10, 10, 0.1)
    y_asph = np.zeros_like(x)
    y_ice = np.zeros_like(x)
    for i in range(x.shape[0]):
        y_asph[i], _ = linear_fiala(x[i], "asphalt")
        y_ice[i], _ = linear_fiala(x[i], "ice")

    plt.figure()
    plt.plot(x, y_asph, label="Asphalt")
    plt.plot(x, y_ice, label="Ice")
    plt.xlabel("Tire Slip Angle (alpha) [deg]")
    plt.ylabel("Lateral Tire Force (F_y) [kN]")
    plt.legend()
    plt.show()
