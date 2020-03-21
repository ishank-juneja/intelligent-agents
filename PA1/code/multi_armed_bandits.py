from scipy.stats import bernoulli, randint
import numpy as np
from bandit_algorithms import *
from bandit_class import MAB
import argparse
import sys


# User Non cmd line constants
# Horizons to sample data at
sample_horizons = [49, 199, 799, 3199, 12799, 51199]
# Whether to record intermediate regret values
#file_list = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
file_list = ["../instances/i-3.txt"]
algos = ['thompson-sampling']
#algos = ['epsilon-greedy']
# For other than eps greedy
eps = 0


if __name__ == '__main__':
    for file in file_list:
        for al in algos:
            for rs in range(20):
                np.random.seed(rs)
                horizon = 205800    # Max out of all horizons, for line printing purpose
                # Read in file containing MAB instance information
                file_in = open(file, 'r')
                probs_str = file_in.readlines()
                # Convert read Bernoulli probability values to floats
                bandit_probs = [float(x) for x in probs_str]
                # Get maximum reward probability among all arms
                p_max = max(bandit_probs)
                # Create MAB class instance for pre determining the possible
                # reward outcomes for given horizon and random seed
                bandit_instance = MAB(bandit_probs, rs, horizon)
                # Determine number of bandits
                n_arms = len(bandit_probs)
                # Initialise cummulative regret and reward
                REW = 0

                if al == 'round-robin':
                    # Round Robin sampling
                    k = 0   # Randomly pick arm 0 to begin with
                    for t in range(horizon):
                        # Add 0/1 reward
                        REW = REW + bandit_instance.sample(k, t)
                        # Pick next arm
                        k = roundRobin(k, n_arms)
                        if t in sample_horizons:
                            sys.stdout.write("{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t+1, p_max*(t+1)-REW))

                elif al == 'epsilon-greedy':
                    for eps in [0.002, 0.02, 0.2]:
                        REW = 0
                        # epsilon greedy sampling
                        # Array of p estimates for all arms
                        # init with value 0.5
                        p_estimates = np.repeat(0.5, n_arms)
                        # Array to record how many times a particular arm sampled
                        nsamps = np.zeros(n_arms)
                        # Pre determine actions to be taken at any
                        # Given time instant t, we can do this since iid
                        X = bernoulli(eps)
                        explore = X.rvs(horizon, random_state=rs)
                        # Pre-determine arm to explore at time t
                        Y = randint(0, n_arms)
                        explore_arm = Y.rvs(horizon, random_state=rs)
                        for t in range(horizon):
                            # Determine arm to be sampled in current step
                            k = epsGreedy(p_estimates, explore[t], explore_arm[t])
                            # Get 0/1 reward
                            r = bandit_instance.sample(k, t)
                            # Update cummulative reward
                            REW = REW + bandit_instance.sample(k, t)
                            # Update p_estimates, compute new empirical mean
                            p_estimates[k] = (nsamps[k]*p_estimates[k] + r)/(nsamps[k] + 1)
                            # Increment number of times kth arm sampled
                            nsamps[k] = nsamps[k] + 1
                            if t in sample_horizons:
                                sys.stdout.write("{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t+1, p_max*(t+1)-REW))
                elif al == 'ucb':
                    # UCB: Upper Confidence Bound Sampling
                    # Array of p estimates for all arms
                    # init with value 0.5
                    p_estimates = np.repeat(0.5, n_arms)
                    # First sample every arm once
                    for t in range(min(n_arms, horizon)):
                        k = t   # Choose arm index same as t
                        r = bandit_instance.sample(k, t)
                        # Update cumm. reward
                        REW = REW + r
                        # Let this reward be p_est
                        p_estimates[k] = r
                        if t in sample_horizons:
                            sys.stdout.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(file, al, rs, eps, t + 1, r))
                            #sys.stdout.write(
                            #    "{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t + 1, p_max * (t + 1) - REW))    
                # Array to record how many times a particular arm sampled
                    nsamps = np.ones(n_arms)    # Each sampled once at start
                    # Now begin UCB based decisions
                    for t in range(n_arms, horizon):
                        # Determine arm to be sampled in current step
                        k = UCB(p_estimates, nsamps, t)
                        # Get 0/1 reward
                        r = bandit_instance.sample(k, t)
                        # Update cummulative reward
                        REW = REW + bandit_instance.sample(k, t)
                        # Update p_estimates, compute new empirical mean
                        p_estimates[k] = (nsamps[k] * p_estimates[k] + r) / (nsamps[k] + 1)
                        # Increment number of times kth arm sampled
                        nsamps[k] = nsamps[k] + 1
                        if t in sample_horizons:
                #sys.stdout.write(
                            #	"{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t + 1, p_max * (t + 1) - REW))
                            sys.stdout.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(file, al, rs, eps, t + 1, r))

                elif al == 'kl-ucb':
                    # KLUCB Sampling algorithm
                    # Array of p estimates for all arms
                    # init with value 0.5
                    p_estimates = np.repeat(0.5, n_arms)
                    # First sample every arm once
                    for t in range(min(n_arms, horizon)):
                        k = t  # Choose arm index same as t
                        r = bandit_instance.sample(k, t)
                        # Update cumm. reward
                        REW = REW + r
                        # Let this reward be p_est
                        p_estimates[k] = r
                        if t in sample_horizons:
                            sys.stdout.write(
                                "{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t + 1, p_max * (t + 1) - REW))
                    # Array to record how many times a particular arm sampled
                    nsamps = np.ones(n_arms)  # Each sampled once at start
                    # Now begin KL-UCB based decisions
                    for t in range(n_arms, horizon):
                        # Determine arm to be sampled in current step
                        k = KL_UCB(p_estimates, nsamps, t)
                        # Get 0/1 reward
                        r = bandit_instance.sample(k, t)
                        # Update cummulative reward
                        REW = REW + bandit_instance.sample(k, t)
                        # Update p_estimates, compute new empirical mean
                        p_estimates[k] = (nsamps[k] * p_estimates[k] + r) / (nsamps[k] + 1)
                        # Increment number of times kth arm sampled
                        nsamps[k] = nsamps[k] + 1
                        if t in sample_horizons:
                            sys.stdout.write(
                                "{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t + 1, p_max * (t + 1) - REW))

                elif al == 'thompson-sampling':
                    # Thompson Sampling algorithm
                    # Array to record how many times a particular arm gave r = 1 (Success)
                    s_arms = np.zeros(n_arms)  # Each sampled once at start
                    # Array to record how many times a particular arm gave r = 0 (Failure)
                    f_arms = np.zeros(n_arms)  # Each sampled once at start
                    # Begin Thompson sampling loop
                    for t in range(horizon):
                        # Determine arm to be sampled in current step
                        k = thompson(s_arms, f_arms)
                        # Get 0/1 reward
                        r = bandit_instance.sample(k, t)
                        # Update cummulative reward
                        REW = REW + bandit_instance.sample(k, t)
                        # kth arm gives success
                        if r == 1:
                            s_arms[k] = s_arms[k] + 1
                        # kth arm gives failure
                        else:
                            f_arms[k] = f_arms[k] + 1
                        if t in sample_horizons:
                            sys.stdout.write(
                                "{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t + 1, p_max * (t + 1) - REW))

                else:
                    print("Invalid algorithm selected")
                    # Don't print REG
                    exit(-1)
