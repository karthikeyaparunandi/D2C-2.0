'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Python class for linearized system identification through MujoCo simulator

Date: July 5, 2019
'''
#!/usr/bin/env python

import numpy as np
import scipy.linalg.blas as blas
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimPool
#import time

class ltv_sys_id_class(object):

	def __init__(self, model_xml_string,  state_size, action_size, n_substeps=1, n_samples=50):

		self.n_x = state_size
		self.n_u = action_size

		#perturbation sigma
		self.sigma = 1e-03
		self.n_samples = n_samples

		self.sim = MjSim(load_model_from_path(model_xml_string), nsubsteps=n_substeps)
		

	def sys_id(self, x_t, u_t, central_diff, activate_second_order=0, V_x_=None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x

		##########################################################################################
		
		XU = np.random.normal(0.0, self.sigma, (self.n_samples, n_x + self.n_u))
		Cov_inv = np.linalg.inv((XU.T) @ XU)
		V_x_F_XU_XU = None

		if central_diff:
			
			F_X_f = simulate((x_t.T) + XU[:, 0:n_x], (u_t.T) + XU[:, n_x:])
			F_X_b = simulate((x_t.T) - XU[:, 0:n_x], (u_t.T) - XU[:, n_x:])
			Y = 0.5*(F_X_f - F_X_b).T
		
		else:

			Y = (simulate((x_t.T) + XU[:, 0:n_x], (u_t.T) + XU[:, n_x:]) - simulate((x_t.T), (u_t.T))).T
			
		F_XU = (Y @ XU) @ Cov_inv

		if activate_second_order:

			assert (central_diff == activate_second_order)
			assert V_x_ is not None

			V_x_ = np.tile(V_x_.T, (self.n_samples, 1))
			
			Z = (F_X_f + F_X_b - 2 * simulate((x_t.T), (u_t.T))).T
			V_x_F_XU_XU = (Cov_inv @ ((XU.T @ (V_x_ @ Z)) @ XU)) @ Cov_inv

		#print(F_XU.shape)
		return F_XU, V_x_F_XU_XU	#(n_samples*self.sigma**2)


	def simulate(self, X, U):
		
		'''
		Function to simulate a batch of inputs given a batch of control inputs and current states
		X - vector of states vertically stacked
		U - vector of controls vertically stacked
		'''
		################## defining local functions & variables for faster access ################

		sim = self.sim
		forward_simulate = self.forward_simulate
		state_output = self.state_output

		##########################################################################################
		
		X_next = []

		# Augmenting X by adding a zero column corresponding to time
		X = np.hstack((np.zeros((X.shape[0], 1)), X))

		for i in range(X.shape[0]):

			X_next.append(state_output(forward_simulate(sim, X[i] , U[i])))


		return np.asarray(X_next)[:,:,0]
	
	def forward_simulate(self, sim, x, u):
		'''
			Function to simulate a single input and a single current state
			Note : THe initial time is set to be zero. So, this can only be used for independent simulations
			x - append time (which is zero here due to above assumption) before state
		'''

		sim.set_state_from_flattened(x)
		sim.forward()
		sim.data.ctrl[:] = u
		sim.step()
		
		return sim.get_state()
			


	def traj_sys_id(self, x_nominal, u_nominal):	
		
		'''
			System identification for a nominal trajectory mentioned as a set of states
		'''
		
		Traj_jac = []
		
		for i in range(u_nominal.shape[0]):
			
			Traj_jac.append(self.sys_id(x_nominal[i], u_nominal[i]))

		return np.asarray(Traj_jac)
		

	
	def state_output(state):

		pass