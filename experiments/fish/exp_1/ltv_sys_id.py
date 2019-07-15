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
		n_u = self.n_u
		##########################################################################################
		
		XU = np.random.normal(0.0, self.sigma, (self.n_samples, n_x + n_u))
		X_ = np.copy(XU[:, :n_x])
		U_ = np.copy(XU[:, n_x:])

		Cov_inv = (XU.T) @ XU
		V_x_F_XU_XU = None

		if central_diff:
			
			F_X_f = simulate((x_t.T) + X_, (u_t.T) + U_)
			F_X_b = simulate((x_t.T) - X_, (u_t.T) - U_)
			Y = 0.5*(F_X_f - F_X_b)
		
		else:

			Y = (simulate((x_t.T) + X_, (u_t.T) + U_) - simulate((x_t.T), (u_t.T))).T

			
		F_XU = np.linalg.solve(Cov_inv, (XU.T @ Y)).T

		if activate_second_order:

			assert (central_diff == activate_second_order)
			assert V_x_ is not None

			
			Z = (F_X_f + F_X_b - 2 * simulate((x_t.T), (u_t.T))).T

			D_XU = self.khatri_rao(XU.T, XU.T)
			
			triu_indices = np.triu_indices((n_x + n_u))
			linear_triu_indices = (n_x+n_u)*triu_indices[0] + triu_indices[1]
			
			D_XU_lin = np.copy(D_XU[linear_triu_indices,:])
			#print(10**10 * D_XU_lin @ D_XU_lin.T)
			V_x_F_XU_XU_ = np.linalg.solve(10**12 * D_XU_lin @ D_XU_lin.T, 10**(12)*D_XU_lin @ (V_x_.T @ Z).T)
			D = np.zeros((n_x+n_u, n_x+n_u))
			# for ind, v in zip(list(np.array(triu_indices).T), V_x_F_XU_XU_):
			# 	V_x_F_XU_XU[ind] = v
			j=0
			for ind in np.array(triu_indices).T:
				D[ind[0]][ind[1]] = V_x_F_XU_XU_[j]
				j += 1
			
			V_x_F_XU_XU = (D + D.T)/2
			
			
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
	
	def vec2symm(self, ):
		pass

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

	def khatri_rao(self, B, C):
	    """
	    Calculate the Khatri-Rao product of 2D matrices. Assumes blocks to
	    be the columns of both matrices.
	 
	    See
	    http://gmao.gsfc.nasa.gov/events/adjoint_workshop-8/present/Friday/Dance.pdf
	    for more details.
	 
	    Parameters
	    ----------
	    B : ndarray, shape = [n, p]
	    C : ndarray, shape = [m, p]
	 
	 
	    Returns
	    -------
	    A : ndarray, shape = [m * n, p]
	 
	    """
	    if B.ndim != 2 or C.ndim != 2:
	        raise ValueError("B and C must have 2 dimensions")
	 
	    n, p = B.shape
	    m, pC = C.shape
	 
	    if p != pC:
	        raise ValueError("B and C must have the same number of columns")
	 
	    return np.einsum('ij, kj -> ikj', B, C).reshape(m * n, p)