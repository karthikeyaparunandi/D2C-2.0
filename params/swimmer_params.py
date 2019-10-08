import numpy as np


horizon = 900
state_dimemsion = 10
control_dimension = 2



Q = 9*np.diag([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
Q_final = 900*np.diag([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
R = .005*np.diag([2, 2])

## Mujoco simulation parameters
# Number of substeps in simulation
ctrl_state_freq_ratio = 1
nominal_init_stddev = 0.1

W_x_LQR = 10*np.eye(state_dimemsion)
W_u_LQR = 2*np.eye(control_dimension)
W_x_LQR_f = 100*np.eye(state_dimemsion)

feedback_samples_no = 40