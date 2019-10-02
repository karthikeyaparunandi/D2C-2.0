import numpy as np


horizon = 30
state_dimemsion = 4
control_dimension = 1



Q = 0*np.array([[1,0,0,0],[0,0.2]])
Q_final = 900*np.array([[1,0],[0,0.1]])
R = .01*np.ones((1,1))




feedback_samples_no = 10
