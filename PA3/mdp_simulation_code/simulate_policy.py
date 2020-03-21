import numpy as np
from MDP_class import MDP
import sys
from scipy.stats import rv_discrete

# Input mdp file name
FILE_IN = '../data-for-test-generation/MDP10.txt'
# Number of lines
nlines = 50000
# File out for trajectory
file1 = '../data/v4.txt'
# File out for true value functions
file2 = '../data/d4.txt'


def policy_eval(mdp, pi):
    # Assuming a single terminal state S-1, remove it from
    # policy evaluation to get full rank A matrix
    # Init coeffcient matrix based on diagonal elements having + 1 term
    states = mdp.nstates
    A = np.identity(states)
    # Assign as per bellman's policy eval equations
    for s in range(states):
        A[s, :] = A[s, :] - mdp.gamma * mdp.f_trans[s, pi[s], :states]
    # Assign right side b vector as sum of T * R terms
    b = np.zeros(states)
    for s in range(states):
        b[s] = np.sum(mdp.f_trans[s, pi[s], :states] * mdp.f_reward[s, pi[s], :states])
    # Check if it is an episodic task, in whic case we already knoe
    # value for terminal state = 0 (enforce it)
    if mdp.type == 'episodic':
        A = A[:-1, :-1]
        b = b[:-1]
    # Solve and return Ax = b
    values = np.linalg.solve(A, b)
    if mdp.type == 'episodic':
        # For last state s = |S| - 1
        values = np.append(values, 0)
    return values


if __name__ == '__main__':
    # Init MDP instance
    mdp = MDP(FILE_IN)
    # Generate a random policy to be followed by the MDP
    policy = np.random.randint(0, mdp.nactions, mdp.nstates)
    # Get a random start state
    start = np.random.randint(mdp.nstates)
    # Pre generate nlines number of random seeds for drawing from transistions
    seeds = np.random.randint(0, 10, nlines)
    # Follow this policy for nlines duration after being born in a random state
    cur_state = start
    fout1 = open(file1, 'w')
    fout1.write("{0}\n{1}\n{2}\n".format(mdp.nstates, mdp.nactions, mdp.gamma))
    for i in range(nlines):
        # Get current action
        cur_action = policy[cur_state]
        # Create a discrete random variable with distribution as per transition T[cur_state][cur_action][s']
        # Generate support points as np array
        support = np.arange(mdp.nstates)
        # Init the discrete latent random variable X
        # Create a custm discrete rv class
        custm = rv_discrete(name='custm',
                            values=(support, mdp.f_trans[cur_state, cur_action, :]))
        # Create instance of above
        X_rv = custm()
        # pre determining the values of X for given horizon and random seed
        next_state = X_rv.rvs(1, random_state=seeds[i])[0]
        cur_rew = mdp.f_reward[cur_state][cur_action][next_state]
        fout1.write("{0}\t{1}\t{2}\n".format(cur_state, cur_action, cur_rew))
        cur_state = next_state

    # Get true value functions for policy
    values = policy_eval(mdp, policy)
    # Print these values to file2
    fout2 = open(file2, 'w')
    for i in range(mdp.nstates):
        fout2.write('{0}\n'.format(values[i]))
