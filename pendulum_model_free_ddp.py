'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Model free DDP method with a simple pendulum experiment in MuJoCo simulator.

Date: July 6, 2019
'''
#!/usr/bin/env python

import numpy as np
from model_free_DDP import DDP
import time
from mujoco_py import load_model_from_path, MjSim, MjViewer
from ltv_sys_id import ltv_sys_id_class
from params.pendulum_params import *
import os


class pendulum_D2C_DDP(DDP, ltv_sys_id_class):
	
	def __init__(self, initial_state, final_state, MODEL_XML, alpha, horizon, state_dimemsion, control_dimension, Q, Q_final, R):
		
		'''
			Declare the matrices associated with the cost function
		'''
		self.Q = Q
		self.Q_final = Q_final
		self.R = R

		DDP.__init__(self, MODEL_XML, state_dimemsion, control_dimension, alpha, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, MODEL_XML, state_dimemsion, control_dimension, n_samples=feedback_samples_no)

	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		return np.concatenate([self.angle_normalize(state.qpos), state.qvel]).reshape(-1, 1)

	def angle_normalize(self, x):
		'''
		Function to normalize the pendulum's angle from [0, Inf] to [-np.pi, np.pi]
		'''
		return -((-x+np.pi) % (2*np.pi)) + np.pi

	def cost(self, x, u):
		'''
			Incremental cost in terms of state and controls
		'''
		return (((x - self.X_g).T @ self.Q) @ (x - self.X_g)) + (((u.T) @ self.R) @ u)
	
	def cost_final(self, x):
		'''
			Cost in terms of state at the terminal time-step
		'''
		return (((x - self.X_g).T @ self.Q_final) @ (x - self.X_g)) 

	def initialize_traj(self, path=None):

		'''
		Initial guess for the nominal trajectory by default is produced by zero controls
		'''
		if path is None:
			
			for t in range(0, self.N):
				self.U_p[t] = np.random.normal(0, 0.001, (self.n_u, 1))#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
			
			np.copyto(self.U_p_temp, self.U_p)
			
			self.forward_pass_sim()
			
			np.copyto(self.X_p_temp, self.X_p)

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE


if __name__=="__main__":

	# Path of the model file
	path_to_model_free_DDP = "/home/karthikeya/Documents/research/model_free_DDP"
	path_to_exp = "/experiments/pendulum/exp_6"
	MODEL_XML = path_to_model_free_DDP + "/models/pendulum.xml"
	path_to_file = path_to_model_free_DDP + path_to_exp + "/pendulum_policy.txt"
	training_cost_data_file = path_to_model_free_DDP + path_to_exp + "/training_cost_data.txt"
	path_to_data = path_to_model_free_DDP + path_to_exp + "/pendulum_D2C_DDP_data.txt"

	with open(path_to_data, 'w') as f:

		f.write("D2C training performed for an inverted pendulum task:\n\n")

		f.write("System details : {}\n".format(os.uname().sysname + "--" + os.uname().nodename + "--" + os.uname().release + "--" + os.uname().version + "--" + os.uname().machine))
		f.write("-------------------------------------------------------------\n")

	# Declare other parameters associated with the problem statement
	
	alpha = .9
	n_iterations = 10
	
	# 15 iterations
	
	'''
	W_x_LQR = 10*np.eye(2)
	W_u_LQR = 2*np.eye(1)
	W_x_LQR_f = 100*np.eye(2)
	'''

	# Declare the initial state and the final state in the problem
	initial_state = np.array([[np.pi-0.1],[0]])
	final_state = np.array([[0], [0]])#np.zeros((2,1))

	# Initiate the above class that contains objects specific to this problem
	D2C_pendulum = pendulum_D2C_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimemsion, control_dimension, Q, Q_final, R)
	
	time_1 = time.time()

	# Run the DDP algorithm
	D2C_pendulum.iterate_ddp(n_iterations)
	
	time_2 = time.time()

	D2C_algorithm_run_time = time_2 - time_1
	
	print("D2C-2 algorithm run time taken: ", time_2 - time_1)
	
	# Save the episodic cost
	with open(training_cost_data_file, 'w') as f:

		for cost in D2C_pendulum.episodic_cost_history:
		
			f.write("%s\n" % cost)

	# Save the obtained policy
	D2C_pendulum.save_policy(path_to_file)

	with open(path_to_data, 'a') as f:

			f.write("\nTotal time taken: {}\n".format(D2C_algorithm_run_time))
			f.write("------------------------------------------------------------------------------------------------------------------------------------\n")

	print(D2C_pendulum.X_p[-1])
	
	# Plot the episodic cost during the training
	D2C_pendulum.plot_episodic_cost_history(save_to_path=path_to_model_free_DDP+path_to_exp+"/episodic_cost_training.png")
	
	# Test the obtained policy
	D2C_pendulum.test_episode(1, path_to_file)
	
	

