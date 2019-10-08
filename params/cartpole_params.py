'''
Cartpole parameters

'''
import numpy as np


state_dimemsion = 4
control_dimension = 1

# Cost parameters for nominal design
Q = 5*np.array([[2, 0, 0, 0],[0, 8, 0, 0],[0, 0, .2, 0],[0, 0, 0, 0.3]])
Q_final = 900*np.array([[3, 0, 0, 0],[0, 10, 0, 0],[0, 0, 3, 0],[0, 0, 0, 3]])
R = .005*np.ones((1,1))

## Mujoco simulation parameters
# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.1
horizon = 30
nominal_init_stddev = 0.01

# Cost parameters for feedback design

W_x_LQR = 2*np.eye(state_dimemsion)
W_u_LQR = 1*np.eye(control_dimension)
W_x_LQR_f = 100*np.eye(state_dimemsion)


# D2C parameters
feedback_samples_no = 25
