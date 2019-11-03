import numpy as np
from windy_gridworld import WindyGridWorld
from policy import eps_greedy
import sys


# declare action space
nactions = int(sys.argv[1])
stochastic = bool(int(sys.argv[2]))
# discount factor
gamma = 1


# Define constants
grid_shape = (7, 10)
# Start position
start = (3, 0)
# Goal position in grid
goal = (3, 7)
# The wind array for wind strengths at different locations
# Size same as number of cols in grid - 10
wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
# Init the grid world object, manages state transitions internally
world = WindyGridWorld(grid_shape, start, goal, wind, stochastic)
# Algorithm parameters
alpha = 0.5 # Learning rate
nepisodes = 1000
# Number of runs over which to average
nruns = 100
# Array to hold number of time steps required for ith episode
time_data = np.zeros((nruns, nepisodes))

if __name__ == '__main__':
	# Run over multiple random instances
	for rs in range(nruns):
		# Declare array to hold Q_values Q(s, a), init with zeros
		Q_values = np.zeros((grid_shape[0], grid_shape[1], nactions))
		# Cummulative time steps
		t = 0
		for i in range(nepisodes):
			# Reset world to start state
			world.reset()
			# Get current row
			cur_row = start[0]
			# Get current col
			cur_col = start[1]
			# Reset episode_ended parameter
			episode_ended = False
			# Get epsilon-greedy action wrt current state
			action = eps_greedy(Q_values, cur_row, cur_col, nactions)
			steps = 0
			while not episode_ended:
				# Get next state and reward
				rew, (next_row, next_col) = world.update_state(action)
				# Get epsilon-greedy action wrt next state
				next_action = eps_greedy(Q_values, next_row, next_col, nactions)
				# Make SARSA(0) update to Q values
				Q_values[cur_row][cur_col][action] += \
					alpha * (rew + gamma * Q_values[next_row][next_col][next_action]
							 - Q_values[cur_row][cur_col][action])
				(cur_row, cur_col) = (next_row, next_col)
				action = next_action
				t += 1
				steps += 1
				# Check if episode has ended
				if next_row == goal[0] and next_col == goal[1]:
					episode_ended = True
			# Once episode ended record data point for current episode
			time_data[rs][i] = t

mean_time_data = np.mean(time_data, axis=0)
for i in range(nepisodes):
	print(mean_time_data[i])

