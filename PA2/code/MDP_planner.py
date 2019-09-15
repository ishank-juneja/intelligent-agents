from pulp import *
import numpy
import argparse
from MDP_class import *
from planning_algos import *


file_name = '/home/ishank/Desktop/CS747/PA2/data/continuing/MDP10.txt'
algo = 'hpi'


# def main(args):
#     # Pass file name to create MDP instance
#     MDP(file_name)


if __name__ == '__main__':
    # Initialise MDP instance
    mdp = MDP(file_name)
    if algo == 'lp':
        values_opt = LPsolver(mdp)
    elif algo == 'hpi':
        values_opt = HPIsolver(mdp)
    else:
        print("Incorrect algo name. Either lp or hpi")
    
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
