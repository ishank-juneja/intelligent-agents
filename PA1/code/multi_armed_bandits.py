import numpy as np
from scipy.stats import bernoulli, randint
from bandit_algorithms import *
from bandit_class import MAB


# User Parameters
bandit_instance_file = "../instances/i-3.txt"
rs = 20
horizon = 10000
eps = 0.1


if __name__ == '__main__':
    # Read in file containing MAB instance information
    file_in = open(bandit_instance_file, 'r')
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

    # Round Robin sampling
    k = 0   # Randomly pick arm 0 to begin with
    # Initialise Cumulative reward (REW)
    REW = 0
    for t in range(horizon):
        # Add 0/1 reward
        REW = REW + bandit_instance.sample(k, t)
        # Pick next arm
        k = roundRobin(k, n_arms)
    REG = p_max*horizon - REW
    print(REG)

    # epsilon greedy sampling
    REW = 0 # Cummulative reward
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
    REG = p_max * horizon - REW
    print(REG)

    # UCB: Upper Confidence Bound Sampling
    REW = 0  # Cummulative reward
    # Array of p estimates for all arms
    # init with value 0.5
    p_estimates = np.repeat(0.5, n_arms)
    # First sample every arm once
    for t in range(n_arms):
        k = t   # Choose arm index same as t
        r = bandit_instance.sample(k, t)
        # Update cumm. reward
        REW = REW + r
        # Let this reward be p_est
        p_estimates[k] = r
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
    REG = p_max * horizon - REW
    print(REG)

    # KLUCB Sampling algorithm
    REW = 0  # Cummulative reward
    # Array of p estimates for all arms
    # init with value 0.5
    p_estimates = np.repeat(0.5, n_arms)
    # First sample every arm once
    for t in range(n_arms):
        k = t  # Choose arm index same as t
        r = bandit_instance.sample(k, t)
        # Update cumm. reward
        REW = REW + r
        # Let this reward be p_est
        p_estimates[k] = r
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
    REG = p_max * horizon - REW
    print(REG)

    # Thompson Sampling algorithm
    # To have a potentially different random seed at every instant
    random_seeds_RV = randint(0, 50)
    # Get horizon samples of this random variable with seed rs
    # This makes the entire random_seeds vector random but deterministic
    # Doing this is not necessary since the distribution of thompson samples
    # changes over time, but this adds additionally layer of randomness
    random_seeds = random_seeds_RV.rvs(horizon, random_state=rs)
    REW = 0  # Cummulative reward
    # Array to record how many times a particular arm gave r = 1 (Success)
    s_arms = np.zeros(n_arms)  # Each sampled once at start
    # Array to record how many times a particular arm gave r = 0 (Failure)
    f_arms = np.zeros(n_arms)  # Each sampled once at start
    # Begin Thompson sampling loop
    for t in range(horizon):
        # Determine arm to be sampled in current step
        k = thompson(s_arms, f_arms, t, random_seeds[t])
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
    REG = p_max * horizon - REW
    print(REG)
