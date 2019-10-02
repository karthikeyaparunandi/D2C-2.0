import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


path_to_folder = "/home/karthikeya/Documents/research/model_free_DDP/experiments/cartpole"

with open(path_to_folder+"/exp_1/training_cost_data.txt", 'r') as f:
	J_1 = np.array([float(line.strip()) for line in f])

with open(path_to_folder+"/exp_2/training_cost_data.txt", 'r') as f:
	J_2 = np.array([float(line.strip()) for line in f])

with open(path_to_folder+"/exp_3/training_cost_data.txt", 'r') as f:
	J_3 = np.array([float(line.strip()) for line in f])

with open(path_to_folder+"/exp_4/training_cost_data.txt", 'r') as f:
	J_4 = np.array([float(line.strip()) for line in f])

with open(path_to_folder+"/exp_5/training_cost_data.txt", 'r') as f:
	J_5 = np.array([float(line.strip()) for line in f])

J = [J_1, J_2, J_3, J_4, J_5]

J_avg = np.mean(J, axis=0)
J_std = np.std(J, axis=0)

#mpl.style.use('default')
mpl.style.use('seaborn')

# Plot the mean and average of the above data w.r.t iterations as follows:
plt.plot(np.array([*range(0, J_avg.shape[0])]), J_avg, linewidth=3)
plt.fill_between(np.array([*range(0, J_avg.shape[0])]), J_avg-J_std, J_avg+J_std, color='C5', alpha=0.5)
plt.xlabel("Training iteration count", fontweight="bold", fontsize=20)
plt.xticks(fontsize=14)
plt.ylabel("Episodic cost", fontweight="bold", fontsize=20)
plt.yticks(fontsize=14)
plt.grid(color='white', linewidth=1.5)
plt.legend(['Mean', 'Standard deviation'], fontsize=20)
#plt.title("Episodic cost (avg. of 5 trials) vs No. of training iterations", fontsize=15)
plt.show()
