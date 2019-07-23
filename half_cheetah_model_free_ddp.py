'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Model free DDP method with a 6-link half_cheetah experiment in MuJoCo simulator.

Date: July 6, 2019
'''
#!/usr/bin/env python

import numpy as np
from model_free_DDP import DDP
import time
from mujoco_py import load_model_from_path, MjSim, MjViewer
from ltv_sys_id import ltv_sys_id_class


class model_free_half_cheetah_6_DDP(DDP, ltv_sys_id_class):
	
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
				self.U_p[t] = np.random.normal(0, 0.05, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
			
			np.copyto(self.U_p_temp, self.U_p)
			
			self.forward_pass_sim()
			
			np.copyto(self.X_p_temp, self.X_p)

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE


if __name__=="__main__":

	# Path of the model file
	path_to_model_free_DDP = "/home/karthikeya/Documents/research/model_free_DDP"
	MODEL_XML = path_to_model_free_DDP + "/models/half_cheetah.xml" 
	path_to_file = path_to_model_free_DDP+"/experiments/half_cheetah/exp_1/half_cheetah_policy.txt"
	training_cost_data_file = path_to_model_free_DDP+"/experiments/half_cheetah/exp_1/training_cost_data.txt"

	# Declare other parameters associated with the problem statement
	horizon = 100
	state_dimemsion = 18
	control_dimension = 6

	Q = 12*np.diag(np.concatenate([[2, .01, .01], np.zeros((15,))]))
	Q_final = 350*np.diag(np.concatenate([[2, .1, .07], np.zeros((15, ))]))
	R = 35*np.diag([2, 2, 2, 2, 2, 2])
	
	alpha = .2
	# Declare the initial state and the final state in the problem
	initial_state = np.zeros((18, 1))
	final_state = np.zeros((18, 1))

	final_state[0] = 4
	
	n_iterations = 40
	# Initiate the above class that contains objects specific to this problem
	half_cheetah = model_free_half_cheetah_6_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimemsion, control_dimension, Q, Q_final, R)

	start_time = time.time()

	# Run the DDP algorithm
	half_cheetah.iterate_ddp(n_iterations)
	
	print("Time taken: ", time.time() - start_time)
	
	# Save the episodic cost
	with open(training_cost_data_file, 'w') as f:
		for cost in half_cheetah.episodic_cost_history:
			f.write("%s\n" % cost)

	# Test the obtained policy
	half_cheetah.save_policy(path_to_file)
	half_cheetah.test_episode(1, path_to_file)

	print(half_cheetah.X_p[-1])
	
	# Plot the episodic cost during the training
	half_cheetah.plot_episodic_cost_history()


