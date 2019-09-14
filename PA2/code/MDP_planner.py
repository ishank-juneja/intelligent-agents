from pulp import *
import numpy
import argparse
from MDP_class import *


file_name = '/home/ishank/Desktop/CS747/PA2/data/continuing/MDP2.txt'


# def main(args):
#     # Pass file name to create MDP instance
#     MDP(file_name)


# Function to return all terminal state candidates
# Very last state is always a candidate as promised in PA2
# So in case of an episodic task return list will be non empty
def get_terminal_states(mdp):
    # Char of a terminal state is looping
    # Transitions to itself with probability 1 irrespective
    # of the action chosen by the policy

    return


if __name__ == '__main__':
    # Intialise MDP instance
    mdp_instance = MDP(file_name)
    # Initialise a PuLP Lp solver minimizer
    value_sum = LpProblem("MDP-Planning", LpMinimize)
    # Array to associate an index with every values(s) s \in S, variable
    values = np.arange(mdp_instance.nstates)
    # Convert above indexing to dictionary form for pulp solver, vars named as Vs_i
    val_dict = LpVariable.dicts("Vs", values)
    # Add objective function (sum) to solver, pulp auto recognises this
    # to be the objective because it is added first
    value_sum += lpSum([val_dict[i] for i in values]), "Sum V(s), forall s in S"
    # Add primary constraints to solver
    value_sum += value_sum[i] >= lpSum()

    # Uncomment after completing
    # # Initialise a parser instance
    # parser = argparse.ArgumentParser()
    #
    # # Add arguments to the parser 1 at a time
    # # This step tells the parser what it should expect before it actually reads the arguments
    # # The --<string name> indicate optional arguments that follow these special symbols
    #
    # # MDP instance file
    # parser.add_argument("--mdp", action="store", dest="file_name", type=str)
    # # algorithm type
    # parser.add_argument("--algorithm", action="store", dest="algo", type=str)
    # # Reads Command line arguments, converts read arguments to the appropriate data type
    # args = parser.parse_args()
    # # Call main function
    # main(args)
