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
Iz = 1

# Distance to front axis from center of mass
a = 1

# Distance to rear axis from center of mass
b = 1

# tire-slip angle to lateral force ratio (for asphalt)
beta = 1.2 

# linearized version of the fiala tire model 
# returns force and gradient at given alpha
def linear_fiala(alpha, road_type):
    if road_type == "asphalt":
        if alpha < -6:
            return 8, 0
        elif alpha > 6:
            return -8, 0
        else:
            return - 1.33 * alpha, -1.33
    elif road_type == "ice":
        if alpha < -1:
            return 1, 0
        elif alpha > 1:
            return -1, 0
        else:
            return - alpha, -1.


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


def linear_dynamics(prev_values):

    Ux, kappa = get_path_traj()
    
    # extract previous values
    Uyo, ro, _, _  = prev_values["state"]
    uo = prev_values["control"]

    ############ LINEARIZATION ################

    alpha_f_bar = (Uyo + a*ro)/ Ux - uo
    alpha_r_bar = (Uyo - b*ro)/ Ux

    fy_f_bar, cf = linear_fiala(alpha_f_bar, "asphalt")
    fy_r_bar, cr = linear_fiala(alpha_r_bar, "asphalt")

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
  
