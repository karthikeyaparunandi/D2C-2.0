import numpy as np


horizon = 800
state_dimemsion = 16
control_dimension = 5

Q = 9*np.diag([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
Q_final = 900*np.diag([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
R = .005*np.diag([2, 2, 2, 2, 2])


## Mujoco simulation parameters
# Number of substeps in simulation
ctrl_state_freq_ratio = 1
nominal_init_stddev = 0.1


feedback_samples_no = 25