import numpy as np
import scipy
from math import log
from scipy.stats import beta


# User constants
# Tolerance for KL-UCB accuracy/convergence
TOL = 0.001


# Helper functions
def KL_div(p, q):
    return p*log(p + TOL/q + TOL) + (1-p)*log((1-p + TOL)/(1-q + TOL))


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


# pestimates are emperical estimate of probabilities
# nsamps is number of times each arm is sampled
def KL_UCB(p_estimates, nsamps, t):
    # Compute Right hand side expression for KL-UCB
    rhs = (log(t) + 3*log(log(t)))/nsamps
    # Init array to store KL-UCb values
    KL_UCBvals = np.zeros_like(p_estimates)
    nbandits = len(p_estimates)
    for i in range(nbandits):
        # For each bandit search for largest value q in
        # [p_estimates[i], 1] that satisfies KLdiv(estimate, q) <= RHS
        # Initialise binary search
        end = 1
        start = p_estimates[i]
        while end - start > TOL:
            mid = (start + end)/2
            KLmid =  KL_div(p_estimates[i], mid)
            KLend = KL_div(p_estimates[i], end)
            # Check if largest value in current interbal is ok
            if KLend <= rhs[i]:
                KL_UCBvals[i] = end
                break
            # Value lies in first half
            elif KLmid > rhs[i]:
                end = mid
                continue
            # Else value lies in 2nd half
            else:
                start = mid
                continue
        # If end point never satisfied constraint
        if KL_UCBvals[i] == 0:
            # Terminal start value
            KL_UCBvals[i] = start
    k = np.argmax(KL_UCBvals)
    return k


# Function to implement thompson sampling algorithm regret minimization
def thompson(s_arms, f_arms, t, rs):
    # Declare list of Beta random variables
    n_arms = len(s_arms)
    # Array to hold observed samples
    samples = np.zeros_like(s_arms)
    # Create a beta random variable for current arm
    # sample the variable and put it in samples array
    for i in range(n_arms):
        # Declare RV
        X = beta(s_arms[i] + 1, f_arms[i] + 1)
        # Sample RV once
        samples[i] = X.rvs(1, random_state=rs)
    # Return the index of the largest sample
    k = np.argmax(samples)
    return k

# Library only
if __name__ == '__main__':
    exit(0)
