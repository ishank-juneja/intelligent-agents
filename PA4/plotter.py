from matplotlib import pyplot as plt
import numpy as np
import sys

# Get input file name
max_samples = int(sys.argv[1])
file_names = sys.argv[2:]

fig = plt.figure()

for file_name in file_names:
	fin = open(file_name, 'r')
	time_data = np.array([float(x) for x in fin.readlines()])
	episode_data = np.arange(1, len(time_data) + 1)
	plt.plot(time_data[:max_samples], episode_data[:max_samples])

out_name = ""
for file_name in file_names:
	out_name += file_name
	out_name += "_"
out_name += "plot.png"

plt.legend(file_names)
plt.xlabel("Cummulative time steps")
plt.ylabel("Episodes Completed")
plt.title("Episodes completed vs. Cummulative time steps")
plt.savefig("results/" + out_name)
plt.close()

fig = plt.figure()

for file_name in file_names:
	fin = open(file_name, 'r')
	time_data = np.array([float(x) for x in fin.readlines()])
	episode_data = np.arange(1, len(time_data) + 1)
	steps_data = np.zeros_like(time_data)
	steps_data[1:] = time_data[:-1]
	steps_data = time_data - steps_data
	plt.yscale("log")
	plt.plot(episode_data[:max_samples], steps_data[:max_samples])

out_name = ""
for file_name in file_names:
	out_name += file_name
	out_name += "_"
out_name += "_steps_plot.png"

plt.legend(file_names)
plt.ylabel("Steps taken to complete episode")
plt.xlabel("Episode Number")
plt.title("Steps taken to complete each episode")
plt.savefig("results/" + out_name)
plt.close()
