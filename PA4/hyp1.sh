#!/bin/sh
echo "Process plain case ... "
python3 source/agent.py 4 1 > plain_sto
echo "Process Kings moves case ... "
python3 source/agent.py 8 1 > kings_sto
echo "Generate plots ... "
python3 plotter.py 500 plain_sto
python3 plotter.py 500 kings_sto
python3 plotter.py 170 plain_sto kings_sto
