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

# tire-slip angle to lateral force ratio (for asphalt)
beta = 1.2 

# net tire slip angle in degrees, force in kN
def calc_lateral_tire_force(angle):
    # This can be done using a nonlinear function or using a linear function
    return - beta * angle


def get_path_traj():
    return U_x, s, kappa

def get_lateral_forces():
    return Fy_r, Fy_f

def get_tire_angles(U_x, U_y, r, delta):
    angle_f = np.arctan2((U_y + a*r) / U_x) - delta
    angle_r = np.arctan2((U_y + b*r) / U_x)
    return angle_f, angle_r


def true_dynamics(state, control):
    delta = control
    U_x, U_y, r, s, e, delta_psi = state
    U_x, s, kappa = get_path_traj()
    # get kappa from where?
    angle_f, angle_r = get_tire_angles(U_x, U_y, r, delta)
    Fx_r, Fy_r = calc_lateral_tire_force(angle_r)
    Fx_f, Fy_f = calc_lateral_tire_force(angle_f)

    # dynamics equations

    U_x = (Fx_f + Fx_r) / m + r*U_y
    U_y_dot = (Fy_f + Fy_r) / m - r*U_x
    r_dot = (a*Fy_f - b * Fy_r) / I_z
    s_dot = U_x - U_y * delta_psi
    e_dot = U_x + U_y * delta_psi
    delta_psi_dot = r - kappa*s_dot


def no_longitud_dynamics(state, control):

    delta = control

    U_y, r, e, delta_psi = state
    U_x, s, kappa, Fx_r, Fx_f = get_path_traj()

    angle_f, angle_r = get_tire_angles(U_x, U_y, r)
    Fy_r = calc_lateral_tire_force(angle_r)
    Fy_f = calc_lateral_tire_force(angle_f)

    U_x_dot = (Fx_f + Fx_r) / m + r* U_y
    U_y_dot = (Fy_f + Fy_r) / m - r*U_x
    r_dot = (a*Fy_f - b * Fy_r) / I_z
    s_dot = U_x - U_y * delta_psi
    e_dot = U_x + U_y * delta_psi
    delta_psi_dot = r - kappa*s_dot

    return 


def linear_dynamics(state, control, prev_values):

    delta = control

    U_y, r, _, _ = state
    _, _, kappa = get_path_traj()
    # Fx_net = - U_y * r * m
    
    ############ LINEARIZATION ################
    U_x0, U_y0, delta_0, Fy_r0, Fy_f0 , r0 = prev_values

    dFyr_dUy = beta * U_x0 / ((U_y0 - b * r0) * (U_y0 - b * r0)) 
    dFyr_dr = - b * dFyr_dUy
    Fy_r = Fy_r0 + dFyr_dUy * (U_y - U_y0) + dFyr_dr * (r - r0)
    # Fy_r = A @ X + B @ U + C
    # where A = [A_Uy A_r A_e A_psi] , B = B_delta 

    A_Fyr = np.array([dFyr_dUy, dFyr_dr, 0, 0])
    B_Fyr = 0
    C_Fyr = Fy_r0 -  dFyr_dUy *  U_y0 -  dFyr_dr * r0
    
    dFyf_dUy = beta * U_x0 / ((U_y0 - a * r0) * (U_y0 - a * r0)) 
    dFyf_dr = - a * dFyr_dUy

    Fy_f = Fy_f0 + dFyf_dUy * (U_y - U_y0) + dFyf_dr * (r - r0) - beta * (delta - delta_0)

    A_Fyf = np.array([dFyf_dUy, dFyf_dr, 0, 0])
    B_Fyf = - beta
    C_Fyf = Fy_f0 -  dFyf_dUy *  U_y0 -  dFyf_dr * r0 + beta * delta_0

    A_Uy = (1/m)*(A_Fyr + A_Fyf) 
    A_Uy[1] -= U_x0
    B_Uy = (1/m)*(B_Fyr + B_Fyf)
    C_Uy = (1/m)*(C_Fyr + C_Fyf)

    A_r = (a * A_Fyr + b * A_Fyf)/ I_z
    B_r = (a * B_Fyr + b * B_Fyf)/ I_z
    C_r = (a * C_Fyr + b * C_Fyf)/ I_z

    A_e = np.array([1, 0, 0, U_x0])
    B_e = 0
    C_e = 0

    A_psi = np.array([0, 1, 0, 0])
    B_psi = 0
    C_psi = - kappa * U_x0

    A = np.array([A_Uy, A_r, A_e, A_psi])
    B = np.array([B_Uy, B_r, B_e, B_psi])
    C = np.array([C_Uy, C_r, C_e, C_psi])

    ################ DYNAMICS #################

    # U_x_dot = (Fx_net) / m + r* U_y
    # U_y_dot = (Fy_f + Fy_r) / m - r*U_x
    # r_dot = (a*Fy_f - b * Fy_r) / I_z
    # s_dot = U_x - U_y * delta_psi
    # e_dot = U_x + U_y * delta_psi
    # delta_psi_dot = r - kappa * s_dot

    ########### INTEGRATE DYNAMICS ##############

    # U_x += U_x_dot * dt 
    # U_y += U_y_dot * dt
    # r += r_dot * dt
    # s += s_dot * dt
    # e += e_dot * dt
    # delta_psi += delta_psi_dot * dt

    # next_state = (U_y, r, e, delta_psi)
    # prev_vals = (U_x, U_y, delta, Fy_r, Fy_f , r)

    return (A, B, C), (Fy_r, Fy_f)
