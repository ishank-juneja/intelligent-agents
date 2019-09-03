from scipy.stats import bernoulli, randint
from bandit_algorithms import *
from bandit_class import MAB
import argparse
import sys


# User Non cmd line constants
# Horizons to sample data at
sample_horizons = [49, 199, 799, 3199, 12799, 51199, 204799]
sample_regrets = []
# Whether to record intermediate regret values
record = True


# Main function to be called as per cmd line arguments
def main(args):
    rs = args.rs
    horizon = args.horizon
    # Read in file containing MAB instance information
    file_in = open(args.bandit_instance_file, 'r')
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
    REG = REW = 0

    if args.al == 'round-robin':
        # Round Robin sampling
        k = 0   # Randomly pick arm 0 to begin with
        for t in range(horizon):
            # Add 0/1 reward
            REW = REW + bandit_instance.sample(k, t)
            # Pick next arm
            k = roundRobin(k, n_arms)
            if record and t in sample_horizons:
                sample_regrets.append(p_max * (t+1) - REW)
        # Final regret value
        REG = p_max*horizon - REW

    elif args.al == 'epsilon-greedy':
        # epsilon greedy sampling
        # Array of p estimates for all arms
        # init with value 0.5
        p_estimates = np.repeat(0.5, n_arms)
        # Array to record how many times a particular arm sampled
        nsamps = np.zeros(n_arms)
        # Pre determine actions to be taken at any
        # Given time instant t, we can do this since iid
        X = bernoulli(args.eps)
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
            if record and t in sample_horizons:
                sample_regrets.append(p_max * (t+1) - REW)
        # Final regret value
        REG = p_max * horizon - REW

    elif args.al == 'ucb':
        # UCB: Upper Confidence Bound Sampling
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
            if record and t in sample_horizons:
                sample_regrets.append(p_max * (t+1) - REW)
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
            if record and t in sample_horizons:
                sample_regrets.append(p_max * (t+1) - REW)
        # Final regret value
        REG = p_max * horizon - REW

    elif args.al == 'kl-ucb':
        # KLUCB Sampling algorithm
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
            if record and t in sample_horizons:
                sample_regrets.append(p_max * (t+1) - REW)
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
            if record and t in sample_horizons:
                sample_regrets.append(p_max * (t+1) - REW)
        REG = p_max * horizon - REW

    elif args.al == 'thompson-sampling':
        # Thompson Sampling algorithm
        # To have a potentially different random seed at every instant
        random_seeds_RV = randint(0, 50)
        # Get horizon samples of this random variable with seed rs
        # This makes the entire random_seeds vector random but deterministic
        # Doing this is not necessary since the distribution of thompson samples
        # changes over time, but this adds additionally layer of randomness
        random_seeds = random_seeds_RV.rvs(horizon, random_state=rs)
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
            if record and t in sample_horizons:
                sample_regrets.append(p_max * (t + 1) - REW)
        REG = p_max * horizon - REW

    else:
        print("Invalid algorithm slected")
        # Don't print REG
        exit(-1)
    # Print run parameters to console
    # If multiple specific sample points were requested
    if record:
        for i in range(len(sample_regrets)):
            sys.stdout.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(args.bandit_instance_file, args.al,
                                                          args.rs, args.eps, sample_horizons[i]+1, sample_regrets[i]))
    # Single terminal value to be printed
    else:
        sys.stdout.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(args.bandit_instance_file, args.al,
                                                      args.rs, args.eps, args.horizon, REG))


if __name__ == "__main__":
    # Initialise a parser instance
    parser = argparse.ArgumentParser()

    # Add arguments to the parser 1 at a time
    # This step tells the parser what it should expect before it actually reads the arguments
    # The --<string name> indicate optional arguments that follow these special symbols

    # Bandit instance file
    parser.add_argument("--instance", action="store", dest="bandit_instance_file", type=str)
    # algorithm type
    parser.add_argument("--algorithm", action="store", dest="al", type=str)
    parser.add_argument("--randomSeed", action="store", dest="rs", type=int)
    # eps value for eps greedy
    parser.add_argument("--epsilon", action="store", dest="eps", type=float)
    parser.add_argument("--horizon", action="store", dest="horizon", type=int)
    # Reads Command line arguments, converts read arguments to the appropriate type
    args = parser.parse_args()
    # Call main function
    main(args)
