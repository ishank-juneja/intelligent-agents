from random import random, randint
import numpy as np


eps = 0.1   # Exploration rate


def eps_greedy(Q_values, cur_row, cur_col, nactions):
    # Observe unifrom random variable between 0 and 1
    rand = random()
    if rand < eps:
        act = randint(0, nactions - 1)
    else:
        act = np.argmax(Q_values[cur_row][cur_col])
    return act