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
		n_substeps = 2

		DDP.__init__(self, MODEL_XML, state_dimemsion, control_dimension, alpha, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, MODEL_XML, state_dimemsion, control_dimension, n_substeps, n_samples=80)

	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		
		return np.concatenate([state.qpos, state.qvel]).reshape(-1, 1)


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
				self.U_p[t] = np.random.normal(0, 0.01, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
			
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
	state_dimemsion = 47
	control_dimension = 17

	Q = 4*np.diag(np.concatenate([[0, 0, 1], np.zeros((44,))]))
	Q_final = 185*np.diag(np.concatenate([[0, 0, 1], np.zeros((44,))]))
	R = 12*np.diag(np.array([1, 1, 1, 1, 1, 3, 2, 1, 1, 3, 2, .25, .25, .25, .25, .25, .25]))
	
	alpha = .2

	# Declare the initial state and the final state in the problem
	initial_state = np.zeros((47, 1))
	final_state = np.zeros((47, 1))
	# initial_state[0] = 
	# initial_state[2] = 0.105
	# initial_state[3] = 1.0
	initial_state = np.concatenate([np.array([-3.52066057e-02,  1.31283831e-04,  8.57946214e-02,  9.99858849e-01,
       -3.48811655e-04,  1.67961900e-02, -2.20250050e-04, -2.02116709e-03,
       -1.11693706e-01,  7.13605021e-04,  8.97091459e-03, -2.06751643e-03,
        9.29697201e-02, -3.36252153e-02, -1.38229401e-02, -2.83350616e-03,
        9.27450419e-02, -3.35980333e-02, -1.95417354e-01,  1.88587742e-01,
       -1.17972357e+00,  2.01410922e-01, -1.82064876e-01, -1.17832641e+00]), np.array([ 2.37867187e-04, -6.21083321e-04, -1.83699986e-04, -1.60525437e-04,
       -2.00597048e-04, -5.42800701e-03,  1.02743736e-02, -1.40029687e-03,
        5.05115545e-03, -9.34712462e-02,  1.42520382e-02, -5.68635245e-03,
       -1.11740777e-02,  2.75866033e-01, -5.12628101e-02,  3.24566181e-04,
       -1.37287466e-02,  3.89309375e-03,  8.18474378e-03, -5.95891325e-03,
        5.13665036e-03,  1.02577187e-02,  8.60547393e-03])]).reshape(-1, 1)
	
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


