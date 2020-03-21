import numpy as np
from time import sleep


# Algorithms that uses 1 step returns to update estimates of values
def TD_zero(nstates, nlines, gamma, trajectory):
    # Choose a constant learning rate independent of index
    alpha = 1e-4
    # Choose a sum squared error change threshold
    err = 5e-4*np.max(trajectory[:, 2])
    # Epoch number
    epoch = 0
    # Max number of iterations
    MAX_INTER = 1000
    # To hold current estimate of values
    values_cur = np.zeros(nstates)
    # To hold previous estimate of values
    values_prev = np.zeros_like(values_cur)
    values_cur[0] = 2*err
    # Make passes over trajectory until desired error goal is reached
    while np.sum(np.abs(values_cur - values_prev)) > err and epoch < MAX_INTER:
        epoch += 1
        # print(np.sum(np.abs(values_cur - values_prev)))
        # Update previous values as being the current at the end of previous iteration
        values_prev[:] = values_cur
        for i in range(nlines):
            cur_state = int(trajectory[i][0])
            next_state = int(trajectory[i + 1][0])
            # Update estimate for the state we have encountered by updating average value
            # and 1 step return boot-strapping of TD(0)
            values_cur[cur_state] = values_cur[cur_state] + \
                alpha * (trajectory[i][2] + gamma*values_cur[next_state] - values_cur[cur_state])
        # print("Epoch number {0}".format(epoch))
    return values_cur


# Algorithms that uses 1 step returns to update estimates of values
def TD_zero2(nstates, nlines, gamma, trajectory):
    # Choose a constant learning rate independent of index
    alpha = 1e-4
    # Choose a sum squared error change threshold
    err = 1e-5
    # Epoch number
    epoch = 0
    # Max number of iterations
    MAX_INTER = 1000
    # To hold current estimate of values
    values_cur = np.zeros(nstates)
    # To hold previous estimate of values
    values_prev = np.zeros_like(values_cur)
    values_cur[0] = 2*err
    # Make passes over trajectory until desired error goal is reached
    while np.sum(np.abs(values_cur - values_prev)) > err and epoch < MAX_INTER:
        epoch += 1
        print(np.sum(np.abs(values_cur - values_prev)))
        # Update previous values as being the current at the end of previous iteration
        values_prev[:] = values_cur
        for i in range(nlines - 1):
            cur_state = int(trajectory[i][0])
            # Next to next
            next_state = int(trajectory[i + 2][0])
            # Update estimate for the state we have encountered by updating average value
            # and 1 step return boot-strapping of TD(0)
            values_cur[cur_state] = values_cur[cur_state] + \
                alpha * (trajectory[i][2] + gamma*trajectory[i+1][2] +
                         gamma*gamma*values_cur[next_state] - values_cur[cur_state])
        print("Epoch number {0}".format(epoch))
    return values_cur


def TD_lambda(nstates, nlines, gamma, trajectory):
    # To old current estimate of values
    values_cur = np.zeros(nstates)
    # Choose a constant learning rate independent of index
    alpha = 1e-6
    # Choose a sum squared error change threshold
    err = 1e-4
    # Epoch number
    epoch = 0
    # Max number of iterations
    MAX_INTER = 2000
    # Init the eligibility trace
    elig = np.zeros(nstates)
    # The forgetting rate lambda
    lamb = 0.95
    # To hold previous estimate of values
    values_prev = np.zeros_like(values_cur)
    values_cur[0] = 2 * err
    # Make passes over trajectory until desired error goal is reached
    while np.sum(np.abs(values_cur - values_prev)) > err and epoch < MAX_INTER:
        epoch += 1
        print(np.sum(np.abs(values_cur - values_prev)))
        # Update previous values as being the current at the end of previous iteration
        values_prev[:] = values_cur
        for i in range(nlines):
            # Get current state and next states
            cur_state = int(trajectory[i][0])
            next_state = int(trajectory[i + 1][0])
            # Compute temporal difference term
            delta = trajectory[i][2] + gamma * values_cur[next_state] - values_cur[cur_state]
            # Update eligibility trace from last time
            elig = elig * lamb * gamma
            elig[cur_state] += 1
            # Update the values of all states as per their eligibilities
            values_cur = values_cur + alpha * delta * elig
        print("Epoch number {0}".format(epoch))
    return values_cur
    # Converge to true value function


def first_visit_mc(nstates, nlines, gamma, trajectory):
    values = np.zeros(nstates)
    # Temporary variable to hold returns in trajectory
    G = 0
    # Iterate over all the lines in the trajectory
    for i in reversed(range(nlines)):
        G = gamma*G + trajectory[i][2]
        cur_state = int(trajectory[i][0])
        # If we have reached the first visit of s_t iterating from back
        if cur_state not in trajectory[:i, 0]:
            values[cur_state] = G
    return values


def every_visit_mc(nstates, nlines, gamma, trajectory):
    values = np.zeros(nstates)
    # Init an array to keep track of number of times a state is
    # visited for monte-carlo averaging
    nvisits = np.zeros(nstates, dtype=int)
    # Temporary variable to hold returns in trajectory
    G = 0
    # Iterate over all the lines in the trajectory
    for i in reversed(range(nlines)):
        G = gamma*G + trajectory[i][2]
        cur_state = int(trajectory[i][0])
        # Append an estimate for the state we have encountered by updating average value
        values[cur_state] = (nvisits[cur_state]*values[cur_state] + G)/(nvisits[cur_state] + 1)
        nvisits[cur_state] = nvisits[cur_state] + 1
    return values