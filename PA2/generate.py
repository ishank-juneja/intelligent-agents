# Taken from Kalpesh Krishna's Github 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/MDP100_0.txt", type=str, help="Output file")
parser.add_argument("--randomseed", default=0, type=int, help="Seed to be used in rpi")
args = parser.parse_args()

np.random.seed(args.randomseed)

S = 100
A = 70

R = np.random.uniform(-1.0, 1.0, [S, A, S])
T = np.random.uniform(0.0, 1.0, [S, A, S])
T = np.divide(T, np.sum(T, axis=2, keepdims=True))

gamma = 0.6

output = ""
output += str(S) + "\n"
output += str(A) + "\n"

for s in range(0, S):
    for a in range(0, A):
        for sPrime in range(0, S):
            output += str(R[s][a][sPrime]) + "\t"
        output += "\n"

# # Reward is zero for terminal state
# for a in range(0, A):
#     for sPrime in range(0, S):
#         output += str(0) + "\t"
#     output += "\n"

for s in range(0, S):
    for a in range(0, A):
        for sPrime in range(0, S):
            output += str(T[s][a][sPrime]) + "\t"
        output += "\n"

# transition is with prob = 1 to itself
# for a in range(0, A):
#     for sPrime in range(0, S):
#         if sPrime == S - 1:
#         	output += str(1) + "\t"
#     	else:
#     		output += str(0) + "\t"
#     output += "\n"

output += str(gamma) + "\n"
output += 'continuing' + '\n'

with open(args.output, 'w') as f:
    f.write(output)
