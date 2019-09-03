import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

x_points = np.array([50, 200, 800, 3200, 12800, 51200, 204800])
y_points = np.zeros_like(x_points)
#LABELS = ['round-robin', 'epsilon-greedy-0.002', 'epsilon-greedy-0.02', 'epsilon-greedy-0.2', 'ucb', 'kl-ucb', 'thompson-sampling']
#LABELS = ['round-robin', 'ucb', 'kl-ucb']
LABELS = ['round-robin', 'epsilon-greedy-0.002', 'epsilon-greedy-0.02', 'epsilon-greedy-0.2', 'ucb', 'kl-ucb']
bandit_data = pd.read_csv('outputData.txt', sep=", ", header=None)
bandit_data.columns = ["file", "algo", "rs", "eps", "horizon", "REG"]
plt.figure(figsize=(6, 6))

data_file1 = bandit_data.loc[bandit_data["file"] == "../instances/i-1.txt"]

# Plot results for file 1
# Round robin 
file1_roundRobin = data_file1.loc[data_file1["algo"] == "round-robin"]
file1_eps = data_file1.loc[data_file1["algo"] == "epsilon-greedy"]
#print(file1_eps)
file1_eps1 = file1_eps.loc[data_file1["eps"] == 0.002] 
file1_eps2 = file1_eps.loc[data_file1["eps"] == 0.02]
file1_eps3 = file1_eps.loc[data_file1["eps"] == 0.2]
file1_UCB = data_file1.loc[data_file1["algo"] == "ucb"]
file1_KLUCB = data_file1.loc[data_file1["algo"] == "kl-ucb"]
#file1_thompson = data_file1.loc[data_file1["algo"] == "thompson-sampling"]

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_roundRobin.loc[file1_roundRobin["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='b')
#plt.plot(np.log10(x_points), y_points, color='b')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps1.loc[file1_eps1["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='r')
#plt.plot(np.log10(x_points), y_points, color='r')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps2.loc[file1_eps2["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='g')
#plt.plot(np.log10(x_points), y_points, color='g')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps3.loc[file1_eps3["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='k')
#plt.plot(np.log10(x_points), y_points, color='k')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_UCB.loc[file1_UCB["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='c')
#plt.plot(np.log10(x_points), y_points, color='c')
# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_KLUCB.loc[file1_KLUCB["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='m')
#plt.plot(np.log10(x_points), y_points, color='m')

# Get data points for each algorithm
#for i in range(len(y_points)):
#	y_points[i] = file1_thompson.loc[file1_thompson["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log(y_points), color='y')

plt.legend(LABELS)

plt.xlabel("Horizon T in log scale")
plt.ylabel("Expected Cumm. Regret, avg. of 50 samp. Log scale")
plt.title("Plot for Instance 1 - y Log scale")
plt.savefig("instance1_log.png")
plt.close()

# Update
data_file1 = bandit_data.loc[bandit_data["file"] == "../instances/i-2.txt"]
# Plot results for file 1
# Round robin 
file1_roundRobin = data_file1.loc[data_file1["algo"] == "round-robin"]
file1_eps = data_file1.loc[data_file1["algo"] == "epsilon-greedy"]
#print(file1_eps)
file1_eps1 = file1_eps.loc[data_file1["eps"] == 0.002] 
file1_eps2 = file1_eps.loc[data_file1["eps"] == 0.02]
file1_eps3 = file1_eps.loc[data_file1["eps"] == 0.2]
file1_UCB = data_file1.loc[data_file1["algo"] == "ucb"]
file1_KLUCB = data_file1.loc[data_file1["algo"] == "kl-ucb"]
#file1_thompson = data_file1.loc[data_file1["algo"] == "thompson-sampling"]

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_roundRobin.loc[file1_roundRobin["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='b')
#plt.plot(np.log10(x_points), y_points, color='b')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps1.loc[file1_eps1["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log10(y_points), color='r')
plt.plot(np.log10(x_points), y_points, color='r')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps2.loc[file1_eps2["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log10(y_points), color='g')
plt.plot(np.log10(x_points), y_points, color='g')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps3.loc[file1_eps3["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log10(y_points), color='k')
plt.plot(np.log10(x_points), y_points, color='k')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_UCB.loc[file1_UCB["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log10(y_points), color='c')
plt.plot(np.log10(x_points), y_points, color='c')
# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_KLUCB.loc[file1_KLUCB["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log10(y_points), color='m')
plt.plot(np.log10(x_points), y_points, color='m')

# Get data points for each algorithm
#for i in range(len(y_points)):
#	y_points[i] = file1_thompson.loc[file1_thompson["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log(y_points), color='y')

plt.legend(LABELS)

plt.xlabel("Horizon T in log scale")
plt.ylabel("Expected Cumm. Regret, avg. of 50 samp. Log scale")
plt.title("Plot for Instance 2 - y Log scale")
plt.savefig("instance2_log.png")
plt.close()

#Update
data_file1 = bandit_data.loc[bandit_data["file"] == "../instances/i-3.txt"]

# Plot results for file 1
# Round robin 
file1_roundRobin = data_file1.loc[data_file1["algo"] == "round-robin"]
file1_eps = data_file1.loc[data_file1["algo"] == "epsilon-greedy"]
#print(file1_eps)
file1_eps1 = file1_eps.loc[data_file1["eps"] == 0.002] 
file1_eps2 = file1_eps.loc[data_file1["eps"] == 0.02]
file1_eps3 = file1_eps.loc[data_file1["eps"] == 0.2]
file1_UCB = data_file1.loc[data_file1["algo"] == "ucb"]
file1_KLUCB = data_file1.loc[data_file1["algo"] == "kl-ucb"]
#file1_thompson = data_file1.loc[data_file1["algo"] == "thompson-sampling"]

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_roundRobin.loc[file1_roundRobin["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='b')
#plt.plot(np.log10(x_points), y_points, color='b')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps1.loc[file1_eps1["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='r')
#plt.plot(np.log10(x_points), y_points, color='r')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps2.loc[file1_eps2["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='g')
#plt.plot(np.log10(x_points), y_points, color='g')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_eps3.loc[file1_eps3["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='k')
#plt.plot(np.log10(x_points), y_points, color='k')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_UCB.loc[file1_UCB["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='c')
#plt.plot(np.log10(x_points), y_points, color='c')
# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = file1_KLUCB.loc[file1_KLUCB["horizon"] == x_points[i]]["REG"].mean()

plt.plot(np.log10(x_points), np.log10(y_points), color='m')
#plt.plot(np.log10(x_points), y_points, color='m')

# Get data points for each algorithm
#for i in range(len(y_points)):
#	y_points[i] = file1_thompson.loc[file1_thompson["horizon"] == x_points[i]]["REG"].mean()

#plt.plot(np.log10(x_points), np.log(y_points), color='y')

plt.legend(LABELS)

plt.xlabel("Horizon T in log scale")
plt.ylabel("Expected Cumm. Regret, avg. of 50 samp. Log scale")
plt.title("Plot for Instance - 3 y Log scale")
plt.savefig("instance3_log.png")
plt.close()
