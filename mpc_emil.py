'''
Den här koden klarar av 50 meter på alla bilar och går tillbaka till initiala
om det inte går att splitta. Det blir aldrig fel.
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy as cp
import warnings
from scipy import sparse

# Vehicle parameters
LENGTH = 12.0  # [m]
WIDTH = 3.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 9.0  # [m]
DBRM = 1.0 # distance between rear and middle [m]

# Road length
ROAD_LENGTH = 200 # [m]

# Initial acceleration
U_INIT = 0.0 # m/s^2

INIT_DIST = 10.0    # Initial distance between the cars

# Initial velocity
V_INIT = 90.0 / 3.6 # 90km/h -> 90/3.6 m/s

# Prediction horizon
PREDICTION_HORIZON = 30

# Constriants
SAFETY_DISTANCE = 1. + LENGTH
MAX_ACCELERATION = 5  # m/s^2
MIN_ACCELERATION = -10 #  m/s^2
MAX_VEL = 120/3.6 # m/s
MIN_VEL = 40/3.6 # m/s

# Max time
MAX_TIME = 200  # [s]

# Time step
DT = 0.2  # [s]

# Air drag coeff
RHO = 1.225 # kg/m^3
CD = 0.8
CD1 = 4.144
CD2 = 7.538

# Vehicle properties
AREA = 6 # m^2
WEIGTH_VEH = 40000.0 # kg
K = RHO*AREA*CD/WEIGTH_VEH

SIZE = 20 # Label size


class VehicleState:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, v=0.0):
        self.x = x
        self.v = v


def update_states(veh_states, control_signals, drag_or_not):
    # updates states for all vehicles
    if drag_or_not == 0:
        # Ideal situation
        for i in range(NUM_CARS):
            veh_states[i].x = veh_states[i].x + veh_states[i].v * DT
            veh_states[i].v = veh_states[i].v + control_signals[-NUM_CARS + i]* DT
    else:
        # Air drag cond.
        for i in range(NUM_CARS):
            veh_states[i].x = veh_states[i].x + veh_states[i].v * DT
            if i != 0:
                CD_distance = CD * (1 - (CD1 / (CD2 + (veh_states[i-1].x - veh_states[i].x))))
                acc_ad = 0.5 * RHO * AREA * (veh_states[i].v ** 2) * CD_distance / WEIGTH_VEH  # Air drag (in terms of acceleration) on vehicle i, which depends on the distance to preeceding vehicle
                veh_states[i].v = veh_states[i].v + (control_signals[-NUM_CARS + i] - acc_ad) * DT
    return veh_states


def mpc(veh_states, xref, split_car, last_ref_in_mpc, split_distance, drag_or_not, hard_split, x_last):
    """
    heavily inspired by https://www.cvxpy.org/tutorial/intro/index.html
    """
    # Even if air drag has been choosen the first mpc needs to calculate with A and B matrices for the ideal case
    Ad, Bd, Dd = create_matrices()  # Create A, B and D matrices for the ideal case
    Q, R = cost_matrices(split_car)
    QN = Q

    x0 = []
    for i in range(NUM_CARS-1):
        x0.append(veh_states[i].x - veh_states[i+1].x)
    for i in range(NUM_CARS):
        x0.append(veh_states[i].v)

    # Create two scalar optimization variables.
    u = cp.Variable((NUM_CARS-1, PREDICTION_HORIZON))
    x = cp.Variable((2*NUM_CARS-1, PREDICTION_HORIZON+1))

    constraints = [x[:,0] == x0] # First constraints needs to be the initial state
    cost = 0.0 # Define the cost function
    for k in range(PREDICTION_HORIZON):
        if type(x_last) != type(None) and drag_or_not == 1:   # The first mpc calculation doesn't have predicted values thus the ideal matrices are been used for the first iteration. Therefore the type is None in the first iteration but never after that.
            Ad, Bd, Dd = create_matrices_linear(x_last[:,k+1])
        cost += cp.quad_form(x[:,k] - xref, Q) + cp.quad_form(u[:,k], R)    # Add the cost function sum(x^TQx + u^TRu)
        constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k] + Dd]             # Add the system x(k+1) = Ad*x(k) + Bd*u(k) + Dd
        for i in range(NUM_CARS):                                           # The for loop is for defining safety distance for all cars
            constraints += [MIN_VEL <= x[NUM_CARS-1+i,k], x[NUM_CARS-1+i,k] <= MAX_VEL]  # Add the velocity constrain just on the velocity inputs in the state vector
            if i != NUM_CARS-1:
                constraints += [SAFETY_DISTANCE <= x[i,k]]  # Add distance constrain on just the distance inputs in the state vector
        if k >= (PREDICTION_HORIZON - last_ref_in_mpc) and hard_split:
            constraints += [[split_distance + LENGTH + 1.] <= x[split_car-1,k]]
        constraints += [MIN_ACCELERATION <= u[:,k], u[:,k] <= MAX_ACCELERATION] # Constarins for the control signal
    cost += cp.quad_form(x[:,PREDICTION_HORIZON] - xref, QN)

    # Form and solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    sol = prob.solve(solver=cp.ECOS)
    warnings.filterwarnings("ignore")
    return u[:,0].value, x.value


def cost_matrices(split_car):
    # Cost matrices

    r_vec = []
    for i in range(NUM_CARS-1):
        r_vec.append(1.0)
        if (i == split_car-2) or (i == split_car-1): # Make the preceeding splitting vehicle's control signal more important(-2) Make the splitting vehicle's control signal(-1) more important
            r_vec[-1] = 10.0
    R = sparse.diags(r_vec)

    q_vec = [0]*(2*NUM_CARS-1)
    for i in range(NUM_CARS):
        if i != NUM_CARS-1:
            q_vec[i] = 10.0
        q_vec[i+NUM_CARS-1] = 1.0
        if i == split_car-1:
            q_vec[i] = 4*10.0
            q_vec[i-1] = 3*10.0
    Q = sparse.diags(q_vec)
    return Q, R


def create_matrices():
    # Define dimentions for A and B matrices
    A = np.zeros((2*NUM_CARS-1,2*NUM_CARS-1))
    B = np.zeros((2*NUM_CARS-1,(NUM_CARS-1)))
    D = np.zeros(2*NUM_CARS-1)

    for i in range(NUM_CARS-1):
        A[i,NUM_CARS-1+i] = 1
        A[i,NUM_CARS+i] = -1

        B[NUM_CARS+i,i] = 1

    # Identity matrix
    I = sparse.eye(2*NUM_CARS-1)

    # Discretize A and B matrices
    Ad = (I + DT*A)
    Bd =  DT*B
    Dd = DT*D
    return Ad, Bd, Dd


def create_matrices_linear(x_pred):
    # Define dimentions for A and B matrices
    A = np.zeros((2*NUM_CARS-1,2*NUM_CARS-1))
    B = np.zeros((2*NUM_CARS-1,(NUM_CARS-1)))
    D = np.zeros(2*NUM_CARS-1)

    for i in range(NUM_CARS-1):
        S, R, Q = deltax_velocity_dependence(x_pred,i)

        A[i,NUM_CARS-1+i] = 1
        A[i,NUM_CARS+i] = -1
        A[NUM_CARS+i,i] = S
        A[NUM_CARS+i,NUM_CARS+i] = R

        B[NUM_CARS+i,i] = 1
        D[NUM_CARS+i] = Q

    # Identity matrix
    I = sparse.eye(2*NUM_CARS-1)

    # Discretize A and B matrices
    Ad = (I + DT*A)
    Bd =  DT*B
    Dd = DT*D
    return Ad, Bd, Dd


def deltax_velocity_dependence(x_pred,i):
    R = -K * x_pred[NUM_CARS+i] * (1 - CD1/(CD2 + x_pred[i])) # i+1 because when it is been sending to this func. the input is i e.g i=0 but we then need for the second vehicle which is i=1 actualy.
    S = -K/2 * (x_pred[NUM_CARS+i] ** 2) * (CD1/((CD2 + x_pred[i]) ** 2))
    Q = -K/2 * (x_pred[NUM_CARS+i]**2) * (x_pred[i] * (CD1/((CD2 + x_pred[i]) ** 2)) + (1 - CD1/(CD2 + x_pred[i])))
    return R, S, Q


def renew_acc(u_list, try_split, next_control_signals):
    u_list.append(0)    # Leader acc. 0 -> vel. const
    for i in range(NUM_CARS-1):
        try:
            u_list.append(next_control_signals[i])
        except:
            u_list.append(0)    # Let the control signal to be 0 if the mpc cannot solve
            try_split = False
    return u_list, try_split


def renew_x_and_v(veh_states, x_list, v_list, distance_list):
    # Appends new states in position, velocity and acceleration list
    for i in range(NUM_CARS):
        v_list.append(veh_states[i].v)
        x_list.append(veh_states[i].x)

    for i in range(NUM_CARS - 1):
        distance_list.append(veh_states[i].x - veh_states[i+1].x - LENGTH)

    return x_list, v_list, distance_list


def check_split(time, split_ready, try_split, hard_split, split_distance, xref, split_car):
    # Check if the prediction horizon can see the split position
    if (time + DT*PREDICTION_HORIZON) >= split_ready and try_split == True:
        last_ref_in_mpc = (time + PREDICTION_HORIZON * DT - split_ready)//DT     # How many of the last prediction horizon should have the hard constrain on the current distance
        if time >= split_ready - 3*DT: # If time is bigger than split time with 1 prediction horizon
            hard_split = False  # Cancel the hard split condition
        if time >= split_ready - 3*DT: # Change the reference so that the hard constraint becomes soft constraint, 2 time steps before split position
            xref[split_car-1] = split_distance + LENGTH     # Fix the reference for the vehicle we want to split. Instead of having it as hard constraint have it like soft constraint
    else:
        xref[split_car-1] = INIT_DIST + LENGTH # Make so that the reference for the splitting vehicle is initial if the split can not be accomplished and we haven't yet seen the split position
        last_ref_in_mpc = -1     # None of the predictione horizon should have the distance as hard constrain
    return xref, int(last_ref_in_mpc), hard_split


def program_is_done(u_list):
    done = True
    for i in range(3*NUM_CARS-1):
        abs(u_list[-3*NUM_CARS+i])
        if abs(u_list[-3*NUM_CARS+i]) < 0.1: # Look element wise
            done = done and True
        else:
            done = done and False
    return done


def pos_list_create():
    global NUM_CARS
    global X_LIST

    NUM_CARS = int(input_control("Number of cars: ",[2,10]))
    X_POS = 0. # Initial position for the lead vehicle in the platoon is assumed to be at 0.0 meter

    X_LIST = [X_POS]
    for i in range(NUM_CARS-1):
        X_POS -= (INIT_DIST + LENGTH)
        X_LIST.append(X_POS)
    return


def split_event_finder():
    # Taking input on where, how and when the split should occur
    split_event = []
    split_event.append(input_control("Split behind vehicle: ",[1,NUM_CARS-1]))
    split_event.append(input_control("Split distance: ", [10, 100]))
    split_event.append(input_control("Split end position give in time: ", [5,3*DT*PREDICTION_HORIZON]))
    split_event.append(input_control('To simulate with air drag press 1 else 0: ',[0,1]))
    return split_event


def input_control(message, limits):     #controls inputs so that they are correct
    try:
        output = int(input(message))
        if limits[0] <= output and output <= limits[1]:
            return output
        else:
            print('Please input a number between',limits[0],'and',limits[1])
            return input_control(message, limits)
    except:
        print('Please input a number between',limits[0],'and',limits[1])
        return input_control(message, limits)


def main():
    pos_list_create()   # Create the initial position list

    initial_veh_states = [] # Initial states for the vehicles. It consists of the position and the velocity of each vehicle
    initial_xref = [0]*(2*NUM_CARS-1)  # The reference state vector at the beginning

    for i in range(NUM_CARS):
        initial_veh_states.append(VehicleState(X_LIST[i], V_INIT)) # The initial vehicle states are the position at the moment and the initial velocity.
        initial_xref[NUM_CARS-1+i] = V_INIT    # The reference velocity

    for i in range(NUM_CARS-1):
        initial_xref[i] = INIT_DIST + LENGTH # The reference distance between the vehicles at the beginning

    split_event = split_event_finder()
    animation(initial_veh_states, split_event, initial_xref)


main()
