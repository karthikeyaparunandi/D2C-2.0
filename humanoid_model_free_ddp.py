'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Model free DDP method with a 6-link humanoid experiment in MuJoCo simulator.

Date: July 6, 2019
'''
#!/usr/bin/env python

import numpy as np
from model_free_DDP import DDP
import time
from mujoco_py import load_model_from_path, MjSim, MjViewer
from ltv_sys_id import ltv_sys_id_class


class model_free_humanoid_DDP(DDP, ltv_sys_id_class):
	
	def __init__(self, initial_state, final_state, MODEL_XML, alpha, horizon, state_dimemsion, control_dimension, Q, Q_final, R):
		
		'''
			Declare the matrices associated with the cost function
		'''
		self.Q = Q
		self.Q_final = Q_final	
		self.R = R
		n_substeps = 5

		DDP.__init__(self, MODEL_XML, state_dimemsion, control_dimension, alpha, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, MODEL_XML, state_dimemsion, control_dimension, n_substeps, n_samples=40)

	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		
		return np.concatenate([state.qpos, state.qvel]).reshape(-1, 1)


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
				self.U_p[t] = np.random.normal(0, 0.0, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
			
			np.copyto(self.U_p_temp, self.U_p)
			
			self.forward_pass_sim()
			
			np.copyto(self.X_p_temp, self.X_p)

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE


if __name__=="__main__":

	# Path of the model file
	path_to_model_free_DDP = "/home/karthikeya/Documents/research/model_free_DDP"
	MODEL_XML = path_to_model_free_DDP+"/models/humanoid.xml" 
	path_to_file = path_to_model_free_DDP+"/experiments/humanoid/exp_1/humanoid_policy.txt"
	training_cost_data_file = path_to_model_free_DDP+"/experiments/humanoid/exp_2/training_cost_data.txt"

	# Declare other parameters associated with the problem statement
	horizon = 600
	state_dimemsion = 47
	control_dimension = 17

	Q = 9*np.diag(np.concatenate([[1, 1, 1, 1], np.zeros((43,))]))
	Q_final = 100*np.diag(np.concatenate([[1, 1, 1, 1], np.zeros((43,))]))
	R = .1*np.diag(np.ones((control_dimension, )))
	
	alpha = 1

	# Declare the initial state and the final state in the problem
	initial_state = np.zeros((47,1))
	final_state = np.zeros((47,1))

	initial_state[2] = 0
	initial_state[6] = 1
	final_state[1] = 2
	final_state[2] = 1.4
	final_state[3] = 1
	
	n_iterations = 50

	# Initiate the above class that contains objects specific to this problem
	humanoid = model_free_humanoid_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimemsion, control_dimension, Q, Q_final, R)

	start_time = time.time()

	# Run the DDP algorithm
	humanoid.iterate_ddp(n_iterations)
	
	print("Time taken: ", time.time() - start_time)
	
	# Save the episodic cost
	with open(training_cost_data_file, 'w') as f:
		for cost in humanoid.episodic_cost_history:
			f.write("%s\n" % cost)

	# Test the obtained policy
	humanoid.save_policy(path_to_file)
	humanoid.test_episode(1, path_to_file)

	print(humanoid.X_p[-1])
	
	# Plot the episodic cost during the training
	humanoid.plot_episodic_cost_history()


