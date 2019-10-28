import numpy as np


# Class for an MDP instance
class MDP:
    # Constructor that reads a text file to
    # parse the MDP instance
    def __init__(self, file_name):
        # Open MDP instance file
        fin = open(file_name, 'r')
        # Read in number of states from line 1 as an int
        self.nstates = int(fin.readline())
        # Read in number of action types from line 2
        self.nactions = int(fin.readline())
        # Read in reward function into a matrix
        # Init reward matrix
        self.f_reward = np.zeros((self.nstates, self.nactions, self.nstates))
        # Read nstates x nactions number of lines
        for i in range(self.nstates * self.nactions):
            s = i // self.nactions
            a = i % self.nactions
            self.f_reward[s][a] = \
                np.fromstring(fin.readline(), dtype=float, sep='\t')
        # Read in Transition function into a matrix
        self.f_trans = np.zeros_like(self.f_reward)
        for i in range(self.nstates * self.nactions):
            s = i // self.nactions
            a = i % self.nactions
            self.f_trans[s][a] = \
                np.fromstring(fin.readline(), dtype=float, sep='\t')
        # Read discount factor
        self.gamma = float(fin.readline())
        # Read Problem type --> continuing or episodic
        self.type = fin.readline()[:-1]

    # Function to return all terminal state candidates
    # Very last state is always a candidate as promised in PA2
    # So in case of an episodic task return list will be non empty
    def get_terminal_states(self):
        # Transitions to itself with probability 1 irrespective
        # of the action chosen by the policy imply terminal state
        lst = []
        for s in range(self.nstates):
            if np.array_equal(self.f_trans[s, :, s], np.ones(self.nactions)):
                lst.append(s)
        return lst
