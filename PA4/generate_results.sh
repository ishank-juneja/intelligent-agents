#!/bin/sh
python3 code/agent.py 4 0 > plain
python3 code/agent.py 8 0 > kings
python3 code/agent.py 8 1 > stochastic
python3 plotter.py 500 plain
python3 plotter.py 500 kings
python3 plotter.py 500 stochastic
python3 plotter.py 170 plain kings stochastic
