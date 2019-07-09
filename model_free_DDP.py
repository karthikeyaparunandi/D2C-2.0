'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Python class for model free DDP method.

Date: July 6, 2019
'''
#!/usr/bin/env python
 #Assumption 1 : costs are quadratic functions

# Optimization library - though no optimization involved, this is used to import DM, jacobian and hessian calculation functionalities from casadi
from __future__ import division

# Numerics
import numpy as np
import math
import time
# Parameters
#import params
import matplotlib.pyplot as plt
from ltv_sys_id import ltv_sys_id_class
import json

class DDP(object):

	def __init__(self, MODEL_XML, n_x, n_u, alpha, horizon, initial_state, final_state):

		self.X_p_0, self.X_g = initial_state, final_state

		self.n_x, self.n_u, self.N = n_x, n_u, horizon
		
		self.alpha = alpha

		
		# Define nominal state trajectory
		self.X_p, self.X_p_temp = np.zeros((self.N, self.n_x, 1)), np.zeros((self.N, self.n_x, 1))
		 
		# Define nominal control trajectory
		self.U_p, self.U_p_temp = np.zeros((self.N, self.n_u, 1)), np.zeros((self.N, self.n_u, 1))
		
		# Define sensitivity matrices
		self.K = np.zeros((self.N, self.n_u, self.n_x))
		self.k = np.zeros((self.N, self.n_u, 1))
		
		self.V_xx = np.zeros((self.N, self.n_x, self.n_x))
		self.V_x = np.zeros((self.N, self.n_x, 1))

		
		# regularization parameter
		self.mu_min = 2
		self.mu = 1000	#10**(-6)
		self.mu_max = 10**(8)
		self.delta_0 = 2
		self.delta = 1	#self.delta_0
		
		self.c_1 = -1e-1
		self.count = 0
		self.episodic_cost_history = []

	def iterate_ddp(self):

		self.initialize_traj()
		
		for j in range(60):	

			if j<300:
				b_pass_success_flag, del_J_alpha = self.backward_pass()
			else:
				b_pass_success_flag, del_J_alpha = self.backward_pass(activate_second_order_dynamics=1)
			if b_pass_success_flag == 1:

				self.regularization_dec_mu()
				f_pass_success_flag = self.forward_pass(del_J_alpha)

				if not f_pass_success_flag:

					print("Forward pass doomed")
					i = 2

					while not f_pass_success_flag:
 
						#print("Forward pass-trying %{}th time".format(i))
						self.alpha = self.alpha*0.99	#simulated annealing
						i += 1
						f_pass_success_flag = self.forward_pass(del_J_alpha)
						print("alpha = ", self.alpha)

			else:

				self.regularization_inc_mu()
				print("This iteration %{} is doomed".format(j))

			if j<5:
				self.alpha = self.alpha*0.9
			#else:
				
				#self.alpha = self.alpha

			self.episodic_cost_history.append(self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N))	
		#print(self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N))
		

	def backward_pass(self, activate_second_order_dynamics=0):

		k_temp, K_temp, V_x_temp, V_xx_temp = np.copy(self.k), np.copy(self.K), np.copy(self.V_x), np.copy(self.V_xx)
		
		self.V_x[self.N-1] = self.l_x_f(self.X_p[self.N-1])	

		np.copyto(self.V_xx[self.N-1], 2*self.Q_final)

		#initialize before forward pass
		del_J_alpha = 0

		for t in range(self.N-1, -1, -1):
			
			if t>0:
				Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.partials_list(self.X_p[t-1], self.U_p[t], self.V_x[t], self.V_xx[t], activate_second_order_dynamics)

			elif t==0:
				Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.partials_list(self.X_p_0, self.U_p[0], self.V_x[0], self.V_xx[0], activate_second_order_dynamics)

			if np.all(np.linalg.eigvals(Q_uu) <= 0):

				print("FAILED! Q_uu is not Positive definite at t=",t)

				b_pass_success_flag = 0

				np.copyto(self.k, k_temp)
				np.copyto(self.K, K_temp)
				np.copyto(self.V_x, V_x_temp)
				np.copyto(self.V_xx, V_xx_temp)
				
				break

			else:

				b_pass_success_flag = 1
				
				# control-limited as follows
				self.k[t], self.K[t] = -(np.linalg.inv(Q_uu) @ Q_u), -(np.linalg.inv(Q_uu) @ Q_ux)
				
				del_J_alpha += -self.alpha*((self.k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((self.k[t].T) @ (Q_uu @ self.k[t]))
				
				if t>0:
					self.V_x[t-1], self.V_xx[t-1] = Q_x + (self.K[t].T) @ (Q_uu @ self.k[t]) + ((self.K[t].T) @ Q_u) + ((Q_ux.T) @ self.k[t]), Q_xx + ((self.K[t].T) @ (Q_uu @ self.K[t])) + ((self.K[t].T) @ Q_ux) + ((Q_ux.T) @ self.K[t])

		self.count += 1

		return b_pass_success_flag, del_J_alpha


	def forward_pass(self, del_J_alpha):

		# Cost before forward pass
		J_1 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)

		np.copyto(self.X_p_temp, self.X_p)
		np.copyto(self.U_p_temp, self.U_p)

		self.forward_pass_sim()
		
		# Cost after forward pass
		J_2 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)

		z = (J_1 - J_2 )/del_J_alpha

		if z < self.c_1:

			np.copyto(self.X_p, self.X_p_temp)
			np.copyto(self.U_p, self.U_p_temp)
	
			f_pass_success_flag = 0
			print("f",z, del_J_alpha, J_1, J_2)

		else:

			f_pass_success_flag = 1

		return f_pass_success_flag



	def partials_list(self, x, u, V_x_next, V_xx_next, activate_second_order_dynamics=0):

		AB, V_x_F_XU_XU = self.sys_id(x, u, V_x_=V_x_next)
		#print(V_x_F_XU_XU[:self.n_x, :self.n_x])
		# print("hi")
		# print(V_x_F_XU_XU[self.n_u:self.n_x + self.n_u, self.n_x:self.n_x + self.n_u])
		# print(V_x_F_XU_XU[self.n_x:self.n_x + self.n_u, self.n_u:self.n_x + self.n_u])
		# print("bye")

		F_x, F_u = np.copy(AB[:, 0:self.n_x]), np.copy(AB[:, self.n_x:])
		
		Q_x, Q_u = self.l_x(x) + ((F_x.T) @ V_x_next), self.l_u(u) + ((F_u.T) @ V_x_next)
		
		Q_xx = 2*self.Q + ((F_x.T) @ (V_xx_next @ F_x)) 
		Q_ux = (F_u.T) @ (V_xx_next @ F_x) 
		Q_uu = 2*self.R + (F_u.T) @ (V_xx_next @ F_u) 

		if(activate_second_order_dynamics):
			Q_xx +=  V_x_F_XU_XU[:self.n_x, :self.n_x]  
			Q_ux +=  0.5*(V_x_F_XU_XU[self.n_x:self.n_x + self.n_u, :self.n_x ] + V_x_F_XU_XU[:self.n_x, self.n_x:self.n_x + self.n_u].T)
			Q_uu +=  V_x_F_XU_XU[self.n_x:self.n_x + self.n_u, self.n_x:self.n_x + self.n_u]

		return Q_x, Q_u, Q_xx, Q_uu, Q_ux

	def forward_pass_sim(self, render=0):
		
		#self.sim.reset()
		#self.sim.set_state(MjSimState(self.sim.get_state().time, np.array([-9*np.pi/10]), np.array([0]), self.sim.get_state().act, self.sim.get_state().udd_state))
		
		self.sim.set_state_from_flattened(np.concatenate([np.array([self.sim.get_state().time]), self.X_p_0.flatten()]))


		for t in range(0, self.N):
			
			self.sim.forward()

			if t==0:

				self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t] 
			
			else:

				self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t] + (self.K[t] @ (self.X_p[t-1] - self.X_p_temp[t-1])) 
			
			self.sim.data.ctrl[:] = self.U_p[t].flatten()
			self.sim.step()
			self.X_p[t] = self.state_output(self.sim.get_state())

			if render:
				self.sim.render(mode='window')

	
	def cost(self, state, control):

		raise NotImplementedError()

	def initialize_traj(self):
		# initial guess for the trajectory
		pass

	def test_episode(self, path=None):
		
		if path is None:
			self.forward_pass_sim(render=1)
		else:
			pass
			# self.sim.set_state_from_flattened(np.concatenate([np.array([self.sim.get_state().time]), self.X_p_0.flatten()]))
			
			# with open(path) as f:
			# 	arrays = [map(float, line.split('\n')) for line in f]
			# #f = open(path)
			# #U = np.loadtxt(path)
			# #U = f.readlines()
			# for t in range(0, self.N):
			# 	print(U[t])
			# 	self.sim.forward()

			# 	self.sim.data.ctrl[:] = np.array(U[t]).flatten()
			# 	self.sim.step()
			# 	#self.X_p[t] = self.state_output(self.sim.get_state())


	def calculate_total_cost(self, initial_state, state_traj, control_traj, horizon):

		#initialize total cost
		cost_total = self.cost(initial_state, control_traj[0])

		for t in range(0, horizon-1):

			cost_total += self.cost(state_traj[t], control_traj[t+1])

		cost_total += self.cost_final(state_traj[horizon-1])

		return cost_total

	def regularization_inc_mu(self):

		# increase mu - regularization 

		self.delta = np.maximum(self.delta_0, self.delta_0*self.delta)

		self.mu = np.maximum(self.mu_min, self.mu*self.delta)

		if self.mu > self.mu_max:
			self.mu = self.mu_max


		print(self.mu)

	def regularization_dec_mu(self):

		# decrease mu - regularization 

		self.delta = np.minimum(1/self.delta_0, self.delta/self.delta_0)

		if self.mu*self.delta > self.mu_min:
			self.mu = self.mu*self.delta

		else:
			self.mu = self.mu_min

	def plot_(self, y, x=None, show=1):

		if x==None:
			
			plt.plot(y)
			plt.show()
		
		else:

			plt.plot(y, x)
			plt.show()

	def plot_episodic_cost_history(self):

		try:
			self.plot_(np.asarray(self.episodic_cost_history).flatten())

		except:

			print("Plotting failed")
			pass

	def save_policy(self, path_to_file):

		U = {}
		for t in range(0, self.N):
			
			U[t] = self.U_p[t]
			

		with open(path_to_file, 'w') as outfile:  
			json.dump(U, outfile)


	def l_x(self, x):

		return 2*self.Q @ (x - self.X_g)

	def l_x_f(self, x):

		return 2*self.Q_final @ (x - self.X_g)

	def l_u(self, u):

		return 2*self.R @ u
