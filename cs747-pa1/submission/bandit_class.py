from scipy.stats import bernoulli


# A class to define a multiarmed bandit instance
class MAB:
    # Parameterised Constructor with instance variable
    # Pass a list of Bandit bernoulli reward probabilities
    # rs = random seed
    # horizon = total iterations
    def __init__(self, bandit_probs, rs, horizon):
        # Assign bandit probabilities
        self.probs = bandit_probs
        # Use these probabilities to initialize that many Bernoulli RVs
        self.bandits = []   # List of Bernoulli RVs
        self.outcomes = []  # List of outcomes of above RVs
        for prob in self.probs:
            X = bernoulli(prob)
            self.bandits.append(X)
            # If 2 bandit arms have the same reward probability
            # then the below sequences will be identical for the 2
            # However this will not affect performance since the
            # Observer can never know this fact, since it can't sample both
            # the arms at a given time instant
            self.outcomes.append(X.rvs(horizon, random_state=rs))

    # Function to implement sampling of the kth bandit arm
    # at the t-th time step
    # Returns 0-1 reward based on the random sequences
    def sample(self, k, t):
        return self.outcomes[k][t]
