import numpy as np
import scipy
from math import log


# Helper functions
def KL_div(p, q):
    return p*log(p/q) + (1-p)*log((1-p)/(1-q))

# A library of policies for MAB problems
# A policy is a mapping from the set of Histories to the
# set of arms (Next arm identification) or a mapping to a
# probability distribution over the set of arms in case
# of randomized algorithms


# Where prev is only info needed from history
# and n_arms (number of arms) is the only info from set of arms
def roundRobin(prev, n_arms):
    next_arm = (prev + 1) % n_arms
    return next_arm


# Explore is a 0/1 array which decides path to take
# p_estimates are the current estimates of probability values
def epsGreedy(p_estimates, explore, arm):
    if explore == 1:
        k = arm
    else:
        # Ties are broken by indexing priority
        # Lower indices get preference if the
        # p_estimate of 2 arms is the same
        k = np.argmax(p_estimates)
    return k


# pestimates are emperical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    ucb = p_estimates + np.sqrt(2*log(t)/nsamps)
    # Ties are broken by index preference
    k = np.argmax(ucb)
    return k


def KL_UCB(p_estimates, nsamps, t):


   return k


# Library only
if __name__ == '__main__':
    exit(0)
