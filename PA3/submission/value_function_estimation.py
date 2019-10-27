import numpy as np
from value_prediction_algos import *

# Path to data source
file_name = "../data/d2.txt"
# Pick value prediction algorithm
al = 'every-visit-mc'

if __name__ == '__main__':
    # Open source file for trajectory data
    fin = open(file_name, 'r')
    # Read in number of states from line 1 as an int
    nstates = int(fin.readline())
    # Read in number of action types from line 2
    nactions = int(fin.readline())
    # Read in discount factor (Recall gamma is not part of the MDP)
    gamma = float(fin.readline())
    # Read in remaining lines into a list
    txt_lines = fin.readlines()
    # Get number of lines including N+1th line for terminal state
    nlines = len(txt_lines)
    # Init numpy array for trajectory nlines - 1 x 3 size
    trajectory = np.zeros((nlines, 3))
    # Iterate over the list except for last line to read trajectory
    for i in range(nlines - 1):
        # Convert a single line string of floats
        # and assign this to a row of trajecectory array
        trajectory[i] = np.fromstring(txt_lines[i], dtype=float, sep='\t')
    # Add last line to trajectory with -1 as action taken and reward
    # float to be consistent with above reading
    trajectory[nlines - 1][0] = float(txt_lines[nlines - 1])
    trajectory[nlines - 1][1] = -1  # Dummy action
    trajectory[nlines - 1][2] = -1  # Dummy reward for taking dummy action
    # Perform value prediction for the given trajectory using al algorithm
    # Init values array
    values = np.zeros(nstates)
    if al == 'first-visit-mc':
        values = first_visit_mc(nstates, nlines - 1, gamma, trajectory)
    elif al == 'every-visit-mc':
        values = every_visit_mc(nstates, nlines - 1, gamma, trajectory)
    else:
        print("Incorrect algorithm choice")
        exit(0)
    # Print the value functions for all states to STDOUT
    for i in range(nstates):
        print(values[i])
