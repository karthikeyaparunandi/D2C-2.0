'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Python class for model free DDP method.

Date: July 6, 2019
'''
#!/usr/bin/env python
 #Assumption 1 : costs are quadratic functions

from __future__ import division

# Numerics
import numpy as np
import time
# Parameters
#import params
import matplotlib.pyplot as plt
from ltv_sys_id import ltv_sys_id_class
import json

class DDP(object):

	def __init__(self, MODEL_XML, n_x, n_u, alpha, horizon, initial_state, final_state):

		self.X_p_0 = initial_state
		self.X_g   = final_state

		self.n_x = n_x
		self.n_u = n_u
		self.N = horizon

		self.alpha = alpha

		
		# Define nominal state trajectory
		self.X_p = np.zeros((self.N, self.n_x, 1))
		self.X_p_temp = np.zeros((self.N, self.n_x, 1))

		# Define nominal control trajectory
		self.U_p  = np.zeros((self.N, self.n_u, 1))
		self.U_p_temp = np.zeros((self.N, self.n_u, 1))

		# Define sensitivity matrices
		self.K = np.zeros((self.N, self.n_u, self.n_x))
		self.k = np.zeros((self.N, self.n_u, 1))
		
		self.V_xx = np.zeros((self.N, self.n_x, self.n_x))
		self.V_x = np.zeros((self.N, self.n_x, 1))

		
		# regularization parameter
		self.mu_min = 1e-3
		self.mu = 1e-3	#10**(-6)
		self.mu_max = 10**(8)
		self.delta_0 = 2
		self.delta = self.delta_0
		
		self.c_1 = -6e-1
		self.count = 0
		self.episodic_cost_history = []

	def iterate_ddp(self, n_iterations):
		
		'''
			Main function that carries out the algorithm at higher level

		'''
		# Initialize the trajectory with the desired initial guess
		self.initialize_traj()
		
		for j in range(n_iterations):	
			#print(j, self.alpha)
			

			b_pass_success_flag, del_J_alpha = self.backward_pass(activate_second_order_dynamics=0)

			if b_pass_success_flag == 1:

				self.regularization_dec_mu()
				f_pass_success_flag = self.forward_pass(del_J_alpha)

				if not f_pass_success_flag:

					#print("Forward pass doomed")
					i = 2

					while not f_pass_success_flag:
 
						#print("Forward pass-trying %{}th time".format(i))
						self.alpha = self.alpha*0.99	#simulated annealing
						i += 1
						f_pass_success_flag = self.forward_pass(del_J_alpha)
						#print("alpha = ", self.alpha)

			else:

				self.regularization_inc_mu()
				print("This iteration %{} is doomed".format(j))

			if j<5:
				self.alpha = self.alpha*0.9
			else:
				self.alpha = self.alpha*0.999

			#print(self.X_p[-1])

			self.episodic_cost_history.append(self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)[0][0])	
			#print(j, self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N))

	def backward_pass(self, activate_second_order_dynamics=0):

		################## defining local functions & variables for faster access ################

		partials_list = self.partials_list
		k = np.copy(self.k)
		K = np.copy(self.K)
		V_x = np.copy(self.V_x)
		V_xx = np.copy(self.V_xx)

		##########################################################################################
		
		V_x[self.N-1] = self.l_x_f(self.X_p[self.N-1])	

		np.copyto(V_xx[self.N-1], 2*self.Q_final)

		#initialize before forward pass
		del_J_alpha = 0

		for t in range(self.N-1, -1, -1):
			
			if t>0:
				Q_x, Q_u, Q_xx, Q_uu, Q_ux = partials_list(self.X_p[t-1], self.U_p[t], V_x[t], V_xx[t], activate_second_order_dynamics)

			elif t==0:
				Q_x, Q_u, Q_xx, Q_uu, Q_ux = partials_list(self.X_p_0, self.U_p[0], V_x[0], V_xx[0], activate_second_order_dynamics)
			#print(Q_uu)
			try:
				# If a matrix cannot be positive-definite, that means it cannot be cholesky decomposed
				np.linalg.cholesky(Q_uu)

			except np.linalg.LinAlgError:
				
				print("FAILED! Q_uu is not Positive definite at t=",t)

				b_pass_success_flag = 0

				# If Q_uu is not positive definite, revert to the earlier values 
				np.copyto(k, self.k)
				np.copyto(K, self.K)
				np.copyto(V_x, self.V_x)
				np.copyto(V_xx, self.V_xx)
				
				break

			else:

				b_pass_success_flag = 1
				
				# update gains as follows
				Q_uu_inv = np.linalg.inv(Q_uu)
				k[t] = -(Q_uu_inv @ Q_u)
				K[t] = -(Q_uu_inv @ Q_ux)

				del_J_alpha += -self.alpha*((k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((k[t].T) @ (Q_uu @ k[t]))
				
				if t>0:
					V_x[t-1] = Q_x + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_ux.T) @ k[t])
					V_xx[t-1] = Q_xx + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_ux) + ((Q_ux.T) @ K[t])


		######################### Update the new gains ##############################################

		np.copyto(self.k, k)
		np.copyto(self.K, K)
		np.copyto(self.V_x, V_x)
		np.copyto(self.V_xx, V_xx)
		
		#############################################################################################

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
			#print("f",z, del_J_alpha, J_1, J_2)

		else:

			f_pass_success_flag = 1

		return f_pass_success_flag



	def partials_list(self, x, u, V_x_next, V_xx_next, activate_second_order_dynamics):	

		################## defining local functions / variables for faster access ################

		n_x = self.n_x
		n_u = self.n_u

		##########################################################################################
		
		AB, V_x_F_XU_XU = self.sys_id(x, u, central_diff=1, activate_second_order=activate_second_order_dynamics, V_x_=V_x_next)
		
		F_x = np.copy(AB[:, 0:n_x])
		F_u = np.copy(AB[:, n_x:])
		#print(F_x, F_u)
		Q_x = self.l_x(x) + ((F_x.T) @ V_x_next)
		Q_u = self.l_u(u) + ((F_u.T) @ V_x_next)

		Q_xx = 2*self.Q + ((F_x.T) @ ((V_xx_next)  @ F_x)) 
		#print(self.mu*np.identity(V_xx_next.shape[0]))
		Q_ux = (F_u.T) @ ((V_xx_next + self.mu*np.eye(V_xx_next.shape[0])) @ F_x)
		#print(V_xx_next)
		Q_uu = 2*self.R + (F_u.T) @ ((V_xx_next + self.mu*np.eye(V_xx_next.shape[0])) @ F_u) 
		#print("modalu:", Q_uu, x, u, "aipoidi")
		#print(F_u.T )
		if(activate_second_order_dynamics):
			#print(V_x_F_XU_XU)
			Q_xx +=  V_x_F_XU_XU[:n_x, :n_x]  
			Q_ux +=  0.5*(V_x_F_XU_XU[n_x:n_x + n_u, :n_x ] + V_x_F_XU_XU[:n_x, n_x: n_x + n_u].T)
			Q_uu +=  V_x_F_XU_XU[n_x:n_x + n_u, n_x:n_x + n_u]

		return Q_x, Q_u, Q_xx, Q_uu, Q_ux



	def forward_pass_sim(self, render=0):
		
		################## defining local functions & variables for faster access ################

		sim = self.sim
		
		##########################################################################################

		sim.set_state_from_flattened(np.concatenate([np.array([self.sim.get_state().time]), self.X_p_0.flatten()]))


		for t in range(0, self.N):
			
			sim.forward()

			if t==0:

				self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t] 
			
			else:

				self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t] + (self.K[t] @ (self.X_p[t-1] - self.X_p_temp[t-1])) 
			
			sim.data.ctrl[:] = self.U_p[t].flatten()
			sim.step()
			#print(sim.get_state())
			self.X_p[t] = self.state_output(sim.get_state())

			if render:
				sim.render(mode='window')
	
	def cost(self, state, control):

		raise NotImplementedError()

	def initialize_traj(self):
		# initial guess for the trajectory
		pass

	def test_episode(self, render=0, path=None):
		
		'''
			Test the episode using the current policy if no path is passed. If a path is mentioned, it simulates the controls from that path
		'''
		
		if path is None:
		
			self.forward_pass_sim(render=1)
		
		else:
		
			self.sim.set_state_from_flattened(np.concatenate([np.array([self.sim.get_state().time]), self.X_p_0.flatten()]))
			
			with open(path) as f:

				Pi = json.load(f)

			for i in range(0, self.N):
				
				self.sim.forward()

				self.sim.data.ctrl[:] = np.array(Pi['U'][str(i)]).flatten() + np.array(Pi['K'][str(i)]) @ (self.state_output(self.sim.get_state()) - np.array(Pi['X'][str(i)]))
				self.sim.step()
				
				if render:
					self.sim.render(mode='window')



	def calculate_total_cost(self, initial_state, state_traj, control_traj, horizon):

		# assign the function to a local function variable
		incremental_cost = self.cost

		#initialize total cost
		cost_total = incremental_cost(initial_state, control_traj[0])
		cost_total += sum(incremental_cost(state_traj[t], control_traj[t+1]) for t in range(0, horizon-1))
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

	def plot_(self, y, save_to_path=None, x=None, show=1):

		if x==None:
			
			plt.figure(figsize=(7, 5))
			plt.plot(y, linewidth=2)
			plt.xlabel('Training iteration count', fontweight="bold", fontsize=12)
			plt.ylabel('Episodic cost', fontweight="bold", fontsize=12)
			#plt.grid(linestyle='-.', linewidth=1)
			plt.grid(color='.910', linewidth=1.5)
			plt.title('Episodic cost vs No. of training iterations')
			if save_to_path is not None:
				plt.savefig(save_to_path, format='png')#, dpi=1000)
			plt.tight_layout()
			plt.show()
		
		else:

			plt.plot(y, x)
			plt.show()

	def plot_episodic_cost_history(self, save_to_path=None):

		try:
			self.plot_(np.asarray(self.episodic_cost_history).flatten(), save_to_path=save_to_path, x=None, show=1)

		except:

			print("Plotting failed")
			pass

	def save_policy(self, path_to_file):

		Pi = {}
		# Open-loop part of the policy
		Pi['U'] = {}
		# Closed loop part of the policy - linear feedback gains
		Pi['K'] = {}
		Pi['X'] = {}

		for t in range(0, self.N):
			
			Pi['U'][t] = np.ndarray.tolist(self.U_p[t])
			Pi['K'][t] = np.ndarray.tolist(self.K[t])
			Pi['X'][t] = np.ndarray.tolist(self.X_p[t])
			
		with open(path_to_file, 'w') as outfile:  

			json.dump(Pi, outfile)


	def l_x(self, x):

		return 2*self.Q @ (x - self.X_g)

	def l_x_f(self, x):

		return 2*self.Q_final @ (x - self.X_g)

	def l_u(self, u):

		return 2*self.R @ u
