import numpy as np


horizon = 500
state_dimemsion = 4
control_dimension = 1

Q = 2*np.diag([1, .05, .01, .01])
Q_final = 1500*np.diag([2, .5, .1, .1])
R = .1*np.diag([2])


ctrl_state_freq_ratio = 1
nominal_init_stddev = 0.4


feedback_samples_no = 20