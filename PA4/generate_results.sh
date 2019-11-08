#!/bin/sh
echo "Process plain case ... "
python3 source/agent.py 4 0 > plain
echo "Process Kings moves case ... "
python3 source/agent.py 8 0 > kings
echo "Process Stochastic wind case ... "
python3 source/agent.py 8 1 > stochastic
echo "Generate plots ... "
python3 plotter.py 500 plain
python3 plotter.py 500 kings
python3 plotter.py 500 stochastic
python3 plotter.py 170 plain kings stochastic
