'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Model free DDP method with a 6-link humanoidstandup experiment in MuJoCo simulator.

Date: July 6, 2019
'''
#!/usr/bin/env python

import numpy as np
from model_free_DDP import DDP
import time
from mujoco_py import load_model_from_path, MjSim, MjViewer
from ltv_sys_id import ltv_sys_id_class


class model_free_humanoidstandup_DDP(DDP, ltv_sys_id_class):
	
	def __init__(self, initial_state, final_state, MODEL_XML, alpha, horizon, state_dimemsion, control_dimension, Q, Q_final, R):
		
		'''
			Declare the matrices associated with the cost function
		'''
		self.Q = Q
		self.Q_final = Q_final	
		self.R = R
		n_substeps = 5

		DDP.__init__(self, MODEL_XML, state_dimemsion, control_dimension, alpha, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, MODEL_XML, state_dimemsion, control_dimension, n_substeps, n_samples=3000)

	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		
		return np.concatenate([state.qpos[:6], state.qpos[7:], state.qvel]).reshape(-1, 1)


	def convert_2_mujoco_state(self, state):
		#print(state.shape)
		#print(np.array([np.sqrt(1 - np.square(state[3:6]).sum())]).shape)
		return np.concatenate([state[:6], np.array([np.sqrt(1 - np.square(state[3:6]).sum())]), state[6:]])

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
				self.U_p[t] = np.random.normal(0, 0.3, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
			
			np.copyto(self.U_p_temp, self.U_p)
			
			self.forward_pass_sim()
			
			np.copyto(self.X_p_temp, self.X_p)

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE


if __name__=="__main__":

	# Path of the model file
	path_to_model_free_DDP = "/home/karthikeya/Documents/research/model_free_DDP"
	MODEL_XML = path_to_model_free_DDP+"/models/humanoidstandup.xml" 
	path_to_file = path_to_model_free_DDP+"/experiments/humanoidstandup/exp_1/humanoidstandup_policy.txt"
	training_cost_data_file = path_to_model_free_DDP+"/experiments/humanoidstandup/exp_1/training_cost_data.txt"

	# Declare other parameters associated with the problem statement
	horizon = 400
	state_dimemsion = 46
	control_dimension = 17

	Q = 100*np.diag(np.concatenate([[0, 0, 1], np.zeros((43,))]))
	Q_final = 2500*np.diag(np.concatenate([[0, 0, 4], np.zeros((43,))]))
	R = 1*np.diag(np.ones((control_dimension, )))
	
	alpha = .9

	# Declare the initial state and the final state in the problem
	initial_state = np.zeros((46, 1))
	final_state = np.zeros((46, 1))

	initial_state[2] = 0.105
	initial_state[3] = 1.0
	
	
	final_state[2] = 1.4
	#final_state[4] = np.sqrt(2)/2
	#final_state[6] = -np.sqrt(2)/2
	
	
	n_iterations = 50

	# Initiate the above class that contains objects specific to this problem
	humanoidstandup = model_free_humanoidstandup_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimemsion, control_dimension, Q, Q_final, R)

	start_time = time.time()

	# Run the DDP algorithm
	humanoidstandup.iterate_ddp(n_iterations)
	
	print("Time taken: ", time.time() - start_time)
	
	# Save the episodic cost
	with open(training_cost_data_file, 'w') as f:
		for cost in humanoidstandup.episodic_cost_history:
			f.write("%s\n" % cost)

	# Test the obtained policy
	humanoidstandup.save_policy(path_to_file)
	humanoidstandup.test_episode(1, path_to_file)

	print(humanoidstandup.X_p[-1])
	
	# Plot the episodic cost during the training
	humanoidstandup.plot_episodic_cost_history()


