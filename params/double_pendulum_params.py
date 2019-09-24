import numpy as np


horizon = 500
state_dimemsion = 6
control_dimension = 1



Q = 0*np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
Q_final = 500*np.array([[1,0,0,0,0,0],[0,10,0,0,0,0],[0,0,1,0,0,0],[0,0,0,.1,0,0],[0,0,0,0,.1,0],[0,0,0,0,0,.1]])
R = .0001*np.ones((1,1))


nominal_init_stddev = 0.1

feedback_samples_no = 50
