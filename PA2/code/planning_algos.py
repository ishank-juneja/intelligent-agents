import numpy as np
from pulp import *


def LPsolver(mdp):
    # Initialise a PuLP Lp solver minimizer
    value_sum = LpProblem("MDP-Planning", LpMinimize)
    # Array to associate an index with every values(s) s \in S, variable
    values = np.arange(mdp.nstates)
    # Convert above indexing to dictionary form for pulp solver, vars named as Vs_i
    val_dict = LpVariable.dicts("Vs", values)
    # Add objective function (sum) to solver, pulp auto recognises this
    # to be the objective because it is added first
    value_sum += lpSum([val_dict[s] for s in values]), "Sum V(s), for all s in S"
    # Add primary constraints to solver in a nested loop
    for s in range(mdp.nstates):
        # One constraint for every action, from class notes
        for a in range(mdp.nactions):
            value_sum += val_dict[s] - lpSum([mdp.f_trans[s][a][s_prime] * (
                    mdp.f_reward[s][a][s_prime] + mdp.gamma * val_dict[s_prime]
            ) for s_prime in values]) >= 0, "Const: Vs_{0}, action-{1}".format(s, a)
    # If the MDP is episodic, find candidate terminal states
    # May be more than one but PA2 guarantees 1
    if mdp.type == "episodic":
        term_lst = mdp.get_terminal_states()
        # Add zero value function constraint when looking
        # ahead from a terminal state
        for term_state in term_lst:
            value_sum += val_dict[term_state] == 0, "Terminal State const. for state {0}".format(term_state)
    # Print formulation to a text file
    value_sum.writeLP("formulation.lp")
    # Invoke pulp solver
    value_sum.solve()

    # If no solution found
    if value_sum.status != 1:
        print("error")
        exit(-1)

    # assign computed optimal values to vector
    values_opt = np.array([value_sum.variables()[s].varValue for s in values])
    return values_opt


def policy_eval(mdp, pi):
    # Init coeffcient matrix
    A = np.identity(mdp.nstates)
    # Assign as per bellman's policy eval equations
    for s in range(mdp.nstates):
        A[s, :] = A[s, :] - mdp.gamma * mdp.f_trans[s, pi[s], :]
    # Assign right side b vector as sum of T * R terms
    b = np.zeros(mdp.nstates)
    for s in range(mdp.nstates):
        b[s] = np.sum(mdp.f_trans[s, pi[s], :] * mdp.f_reward[s, pi[s], :])
    # Solve and return Ax = b
    return np.linalg.solve(A, b)


def get_max_action_value(mdp, values):
    Q_pi = np.zeros((mdp.nstates, mdp.nactions))
    for s in range(mdp.nstates):
        for a in range(mdp.nactions):
            Q_pi[s, a] = np.sum(mdp.f_trans[s, a, :] * (mdp.f_reward[s, a, :] +
                                                        mdp.gamma * values))
    # Return the maximizers of the action value function over the actions a
    Q_max = np.zeros_like(values, dtype=int)
    for s in range(mdp.nstates):
        # Action that maximizes Q for given pi
        Q_max[s] = np.argmax(Q_pi[s, :])
    return Q_max


def HPIsolver(mdp):
    # Initialise a random prev and current policy vector
    pi_prev = np.random.randint(0, mdp.nactions, mdp.nstates)
    pi_cur = np.copy(pi_prev)
    # Change 1 action in pi_cur to enter loop
    if pi_cur[0] != 0:
        pi_cur[0] = 0
    else:
        pi_cur[0] = 1
    # Init values array
    values = np.zeros_like(pi_prev)
    # Begin policy iteration/improvement loop
    while not np.array_equal(pi_prev, pi_cur):
        # Update pi_prev to pi_cur
        pi_prev = pi_cur
        # Get current performance
        values = policy_eval(mdp, pi_prev)
        # Attempt to improve policy by evaluating action value functions
        pi_cur = get_max_action_value(mdp, values)
    return pi_cur, values
