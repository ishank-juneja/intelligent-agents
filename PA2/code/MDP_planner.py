import argparse
from MDP_class import *
from planning_algos import *
import sys


def print_policy(val_opt, pi_opt):
    for i in range(len(val_opt)):
        sys.stdout.write("{0:.6f}\t{1}\n".format(val_opt[i], pi_opt[i]))
    return


def main(cmd_args):
    # Initialise MDP instance
    mdp = MDP(cmd_args.file_name)
    if cmd_args.algo == 'lp':
        val_opt, pi_opt = LPsolver(mdp)
    elif cmd_args.algo == 'hpi':
        val_opt, pi_opt = HPIsolver(mdp)
    else:
        val_opt = 0
        pi_opt = 0
        print("Incorrect algo name. Either lp or hpi")
    print_policy(val_opt, pi_opt)
    return


if __name__ == '__main__':
    # Initialise a parser instance
    parser = argparse.ArgumentParser()

    # Add arguments to the parser 1 at a time
    # This step tells the parser what it should expect before it actually reads the arguments
    # The --<string name> indicate optional arguments that follow these special symbols

    # MDP instance file
    parser.add_argument("--mdp", action="store", dest="file_name", type=str)
    # algorithm type
    parser.add_argument("--algorithm", action="store", dest="algo", type=str)
    # Reads Command line arguments, converts read arguments to the appropriate data type
    args = parser.parse_args()
    # Call main function
    main(args)
