'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Model free DDP method with a simple pendulum experiment in MuJoCo simulator.

Date: July 6, 2019
'''
#!/usr/bin/env python

import numpy as np
import gym
from model_free_DDP import DDP
import time
from mujoco_py import load_model_from_path, MjSim, MjViewer
from casadi import *
from ltv_sys_id import ltv_sys_id_class
import copy


class pendulum_D2C_DDP(DDP, ltv_sys_id_class):
	
	def __init__(self, initial_state, final_state, MODEL_XML, horizon, state_dimemsion, control_dimension, Q, Q_final, R):
		
		'''
			Declare the matrices associated with the cost function
		'''
		self.Q = Q
		self.Q_final = Q_final
		self.R = R

		DDP.__init__(self, MODEL_XML, state_dimemsion, control_dimension, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, MODEL_XML, state_dimemsion, control_dimension, n_samples=10)

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
		return (mtimes(mtimes((x - self.X_g).T , self.Q) , (x - self.X_g))) + mtimes(mtimes((u.T) , self.R) , u)
	
	def cost_final(self, x):
		'''
			Cost in terms of state at the terminal time-step
		'''
		return (mtimes(mtimes((x - self.X_g).T , self.Q_final) , (x - self.X_g)))

	def initialize_traj(self, path=None):

		'''
		Initial guess for the nominal trajectory by default is produced by zero controls
		'''
		if path is None:
			
			for t in range(0, self.N):
				self.U_p[t] = np.zeros((self.n_u, 1))#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
				self.U_p_temp[t] = copy.deepcopy(self.U_p[t])
				self.X_p[t] = self.X_p_0
				self.X_p_temp[t] = self.X_p_0

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE


if __name__=="__main__":

	# Path of the model file
	MODEL_XML = "/home/karthikeya/Documents/research/DDPG_D2C/libraries/gym/gym/envs/mujoco/assets/pendulum.xml"
	
	# Declare other parameters associated with the problem statement
	horizon = 30
	state_dimemsion = 2
	control_dimension = 1

	Q = 9*np.array([[1,0],[0,0.2]])
	Q_final = 900*np.array([[1,0],[0,0.1]])
	R = 1*np.ones((1,1))
	
	'''
	W_x_LQR = 10*np.eye(2)
	W_u_LQR = 2*np.eye(1)
	W_x_LQR_f = 100*np.eye(2)
	'''
	# Declare the initial state and the final state in the problem
	initial_state = np.array([[0],[0]])
	final_state = np.array([[np.pi/2], [0]])#np.zeros((2,1))

	# Initiate the above class that contains objects specific to this problem
	D2C_pendulum = pendulum_D2C_DDP(initial_state, final_state, MODEL_XML, horizon, state_dimemsion, control_dimension, Q, Q_final, R)

	start_time = time.time()

	# Run the DDP algorithm
	D2C_pendulum.iterate_ddp()
	
	print("Time taken: ", time.time() - start_time)
	
	# Test the obtained policy
	D2C_pendulum.test_episode()

	print(D2C_pendulum.X_p)
	
	# Plot the episodic cost during the training
	D2C_pendulum.plot_episodic_cost_history()


