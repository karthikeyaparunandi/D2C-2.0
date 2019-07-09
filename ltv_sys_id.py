'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Python class for linearized system identification through MujoCo simulator

Date: July 5, 2019
'''
#!/usr/bin/env python

import numpy as np
import scipy.linalg.blas as blas
from mujoco_py import load_model_from_path, MjSim, MjViewer


class ltv_sys_id_class(object):

	def __init__(self, model_xml_string,  state_size, action_size, n_samples=50):

		self.n_x, self.n_u = state_size, action_size

		#perturbation sigma
		self.sigma, self.n_samples = 1e-03, n_samples
		model = load_model_from_path(model_xml_string)
		self.sim = MjSim(model, nsubsteps=1)
		

	def sys_id(self, x_t, u_t, central_diff=1, activate_second_order=1, V_x_=None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		
		XU = np.random.normal(0.0, self.sigma, (self.n_samples, self.n_x + self.n_u))
		Cov_inv = np.linalg.inv((XU.T) @ XU)
		V_x_F_XU_XU = None


		if central_diff:
			
			F_X_f = self.simulate((x_t.T) + XU[:, 0:self.n_x], (u_t.T) + XU[:, self.n_x:])
			F_X_b = self.simulate((x_t.T) - XU[:, 0:self.n_x], (u_t.T) - XU[:, self.n_x:])

			Y = 0.5*(F_X_f - F_X_b).T
		
		else:

			Y = (self.simulate((x_t.T) + XU[:, 0:self.n_x], (u_t.T) + XU[:, self.n_x:]) - self.simulate((x_t.T), (u_t.T))).T
		
		F_XU = (Y @ XU)@ Cov_inv

		if activate_second_order:

			assert (central_diff == activate_second_order)
			assert V_x_ is not None

			V_x_ = np.tile(V_x_.T, (self.n_samples, 1))
			print(V_x_)
			Z = (F_X_f + F_X_b - 2 * self.simulate((x_t.T), (u_t.T))).T
			V_x_F_XU_XU = 2 * (Cov_inv @ ((XU.T @ (V_x_ @ Z)) @ XU)) @ Cov_inv

		#print(F_XU_XU.shape)
		return F_XU, V_x_F_XU_XU	#(n_samples*self.sigma**2)


	def simulate(self, X, U):
		
		'''
		X - vector of states vertically stacked
		U - vector of controls vertically stacked
		'''

		X_next = []
		old_state = self.sim.get_state()
		
		for i in range(X.shape[0]):

			self.sim.set_state_from_flattened(np.concatenate((np.array([old_state.time]), X[i])))
			self.sim.forward()
			self.sim.data.ctrl[:] = U[i]
			self.sim.step()
			old_state = self.sim.get_state()
			X_next.append(self.state_output(old_state))
			
		return np.asarray(X_next)[:,:,0]
	
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