import numpy as np
from time import sleep


# Constants
MAX_LINES = 50000


def first_visit_mc(nstates, nlines, gamma, trajectory):
    values = np.zeros(nstates)
    # Temporary variable to hold returns in trajectory
    G = 0
    # Iterate over all the lines in the trajectory
    for i in reversed(range(nlines)):
        G = gamma*G + trajectory[i][2]
        # print(G)
        cur_state = int(trajectory[i][0])
        # If we have reached the first visit of s_t iterating from back
        if cur_state not in trajectory[:i, 0]:
            # print(cur_state)
            values[cur_state] = G
    return values


def every_visit_mc(nstates, nlines, gamma, trajectory):
    values = np.zeros(nstates)
    # Init an array to keep track of number of times a state is
    # visited for monte-carlo averaging
    nvisits = np.zeros(nstates, dtype=int)
    # Init an array to hold all possible estimates of value function
    # for a given state value using MAXLINES in instance
    estimates = np.zeros((nstates, MAX_LINES))
    # Temporary variable to hold returns in trajectory
    G = 0
    # Iterate over all the lines in the trajectory
    for i in reversed(range(nlines)):
        G = gamma*G + trajectory[i][2]
        # print(G)
        cur_state = int(trajectory[i][0])
        # print(cur_state)
        # Append an estimate for the state we have encountered
        estimates[cur_state][nvisits[cur_state]] = G
        nvisits[cur_state] = nvisits[cur_state] + 1
        # sleep(1)
    for i in range(nstates):
        values[i] = np.sum(estimates[i])/nvisits[i]
    return values


def TD_zero(nstates):
    values = np.zeros(nstates)
    return values


def TD_lambda(nstates):
    values = np.zeros(nstates)
    return values

