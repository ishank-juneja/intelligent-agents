# To produce all the required plots run, from current directory:
$ ./generate_results.sh


# To produce the plots corresponding to a special case, run 
$ python3 source/agent.py <number of moves> <Whether stochastic> > out_file_name
$ python3 plotter.py <no. of episdoes for which to plot> out_file_name
# Here <number of moves> lies in 4 (vanilla), 8 (kings) and 9 (kings + remain in place)
# Whether stochastic is a boolean variable which is 1 in case of stochastic wind and 0 otherwise
# <no. of episdoes for which to plot> is between 1 and 1000, is the number of epiodes for which plots ae generated
