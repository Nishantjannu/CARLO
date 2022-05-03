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

# Mass of vehicle
m = 1

# Yaw inertial of vehicle
I_z = 1

# Distance to front axis from center of mass
a = 1

# Distance to rear axis from center of mass
b = 1


def calc_lateral_tire_force(alpha, tire="rear"):
    # This can be done using a nonlinear function or using a linear function
    if tire == "rear":
        return alpha * 2
    elif tire == "front":
        return alpha * -1.5 + 5


def get_path_traj():
    return U_x, s, kappa


def get_alphas(U_x, U_y, r, delta):
    alpha_f = np.arctan((U_y + a*r) / U_x) - delta
    alpha_r = np.arctan((U_y + b*r) / U_x)


def true_dynamics(state, control):
    delta = control
    U_x, U_y, r, s, e, delta_psi = state
    # U_x, s, kappa = get_path_traj()
    # get kappa from where?
    alpha_f, alpha_r = get_alphas(U_x, U_y, r, delta)
    Fx_r, Fy_r = calc_lateral_tire_force(alpha, "rear")
    Fx_f, Fy_f = calc_lateral_tire_force(alpha, "front")

    U_x = (Fx_f + Fxr) / m + r*U_y
    U_y_dot = (Fy_f + Fy_r) / m - r*U_x
    r_dot = (a*Fy_f - b * Fy_r) / I_z
    s_dot = U_x - U_y * delta_psi
    e_dot = U_x + U_y * delta_psi
    delta_psi_dot = r - kappa*s_dot


def no_longitud_dynamics(state, control):
    delta = control
    U_y, r, e, delta_psi = state
    U_x, s, kappa, Fx_r, Fx_f = get_path_traj()
    alpha_f, alpha_r = get_alphas(U_x, U_y, r, delta)
    Fy_r = calc_lateral_tire_force(alpha, "rear")
    Fy_f = calc_lateral_tire_force(alpha, "front")

    U_x = (Fx_f + Fxr) / m + r*U_y
    U_y_dot = (Fy_f + Fy_r) / m - r*U_x
    r_dot = (a*Fy_f - b * Fy_r) / I_z
    s_dot = U_x - U_y * delta_psi
    e_dot = U_x + U_y * delta_psi
    delta_psi_dot = r - kappa*s_dot


def linear_dynamics():
    pass
