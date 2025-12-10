import numpy as np
from sklearn.linear_model import LinearRegression

# DAG helpers
def contains_cycle(adj):
    eigenvalues = np.linalg.eigvals(adj)
    return np.any(np.abs(eigenvalues) > 1e-10)


def random_dag(p, prob_connect):
    adj = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            if np.random.rand() < prob_connect:
                adj[i, j] = 1
    return adj

# stores a state of algorithm
class GDSState:
    def __init__(self, n, p, adj):
        self.n = n
        self.p = p
        self.adj = adj.copy() # binary adjacency graph
        self.B = np.zeros((p, p)) # edge weights
        self.each_res_var = np.zeros(p) # variances of individual N
        self.sum_res_var = 0.0 # total variance used in score calculation
        self.num_pars = 0 # number of edges
        self.score = np.inf # current score


def initialize_state(X, initial_adj, SigmaHat, pen_factor):
    n, p = X.shape
    state = GDSState(n, p, initial_adj)

    # Calculate MLE variances of variables
    for node in range(p):
        parents = np.where(state.adj[:, node] == 1)[0]
        # If no parents, variance is simply the variance of the data
        if len(parents) == 0:
            state.each_res_var[node] = ((n - 1) / n) * np.var(X[:, node], ddof=1)
        else:
            # In the case of parents, compute variance of the residuals after subtracting off impact of parent variables
            model = LinearRegression()
            model.fit(X[:, parents], X[:, node])
            residuals = X[:, node] - model.predict(X[:, parents])
            state.each_res_var[node] = ((n - 1) / n) * np.var(residuals, ddof=1)
            state.B[node, parents] = model.coef_

    # Calculate sum of variances, number of edges, and score of new state
    state.sum_res_var = np.sum(state.each_res_var)
    state.num_pars = np.sum(state.B != 0) + 1
    state.score = compute_score(state, SigmaHat, pen_factor)
    return state

# Computes score as done in paper
def compute_score(state, SigmaHat, pen_factor):
    n = state.n
    p = state.p
    sigma_hat_sq = (n * state.sum_res_var) / (p * n - 1)
    I = np.eye(p)
    B = state.B
    # Calculate negative log-likelihood
    neg_log_lik = (
        n * p / 2 * np.log(2 * np.pi * sigma_hat_sq) +
        n / (2 * sigma_hat_sq) * np.trace((I - B).T @ (I - B) @ SigmaHat)
    )
    # Penalize edge amount 
    penalization = pen_factor * (np.log(n) / 2) * state.num_pars
    return neg_log_lik + penalization


def compute_new_state(old_state, index, X, SigmaHat, pen_factor):
    n, p = X.shape
    i, j = index

    # Initialize a new state identical to the past one
    new_state = GDSState(old_state.n, old_state.p, old_state.adj)
    new_state.B = old_state.B.copy()
    new_state.each_res_var = old_state.each_res_var.copy()
    new_state.sum_res_var = old_state.sum_res_var
    new_state.num_pars = old_state.num_pars

    # Flip one edge
    new_state.adj[i, j] = (new_state.adj[i, j] + 1) % 2

    # If edge didn't exist and now does, make sure there isn't an edge facing the other direction
    if new_state.adj[j, i] == 1:
        new_state.adj[j, i] = 0
        # If we just reversed an edge, both nodes have their parents change
        recompute_nodes = [i, j]
    else:
        # If we added an edge, only the destination has its parents change
        recompute_nodes = [j]

    # For edges with different parents than before, 
    for node in recompute_nodes:
        new_state.B[node, :] = 0
        old_parents = np.where(old_state.adj[:, node] == 1)[0]
        parents = np.where(new_state.adj[:, node] == 1)[0]

        # Remove old contribution of recalculated node from summary variables
        new_state.sum_res_var -= new_state.each_res_var[node]
        new_state.num_pars -= len(old_parents)

        # Recompute variance of a node that needs recalculation, again using residuals if it has parents
        if len(parents) == 0:
            new_state.each_res_var[node] = ((n - 1) / n) * np.var(X[:, node], ddof=1)
        else:
            model = LinearRegression()
            model.fit(X[:, parents], X[:, node])
            residuals = X[:, node] - model.predict(X[:, parents])
            new_state.each_res_var[node] = ((n - 1) / n) * np.var(residuals, ddof=1)
            new_state.B[node, parents] = model.coef_

        # Readd to summary variables newly calculated values
        new_state.sum_res_var += new_state.each_res_var[node]
        new_state.num_pars += len(parents)

    # Compute score of new state
    new_state.score = compute_score(new_state, SigmaHat, pen_factor)
    return new_state


def one_step_greedy(state_old, X, SigmaHat, k, pen_factor):
    p = state_old.p
    # Compute normalized variance of each node, to prioritize edges facing into high-variance nodes
    var_norm = state_old.each_res_var - np.min(state_old.each_res_var)
    var_norm = var_norm + np.min(var_norm[var_norm > 0]) if np.any(var_norm > 0) else var_norm + 1e-10
    var_norm = var_norm / np.sum(var_norm)
    # Obtain weighted random ordering of nodes
    sort_nodes = np.random.choice(p, size=p, p=var_norm, replace=False)

    # Create our array of possible edge updates, prioritizing edges into earlier entries in sort_nodes
    index_list = []
    for i in range(p):
        for j in range(p):
            if j == i:
                continue
            index_list.append((sort_nodes[j], sort_nodes[i]))
    index_list = np.array(index_list)

    best_state = state_old
    made_step = False
    index_count = 0
    tried = 0

    # As long as we have more moves to make, and either we have not yet made a move or we have not tried k states, continue
    while (not made_step or tried < k) and index_count < len(index_list):
        # Get next move
        i, j = index_list[index_count]
        index_count += 1

        # Flip edges in order determined by variance weighting 
        candidate_adj = state_old.adj.copy()
        candidate_adj[i, j] = (candidate_adj[i, j] + 1) % 2
        candidate_adj[j, i] = 0

        # If the new graph is still a valid acyclic DAG, we compute its properties
        if not contains_cycle(candidate_adj):
            tried += 1
            new_state = compute_new_state(state_old, (i, j), X, SigmaHat, pen_factor)
            # If the new graph has a better BIC than the old one, we record it as the best state so far and record that we have made a step
            if new_state.score < best_state.score:
                best_state = new_state
                made_step = True

    return {'State': best_state, 'madeStep': made_step}


def gds(X):
    # Compute necessary inputs
    n, p = X.shape
    pen_factor = 1 # Available to set extra weighting on edge penalty. 1 gives the BIC, higher numbers penalize edges more
    SigmaHat = np.cov(X, rowvar=False)
    k_vec = [1 * p, 2 * p, 3 * p, 5 * p, 300] # k values suggested in paper to avoid local optima
    states = []

    # For each k-value, run greedy optimization checking at least k states until no move is made
    for k in k_vec:
        initial_adj = random_dag(p, 2 / (p - 1))
        state = initialize_state(X, initial_adj, SigmaHat, pen_factor)

        made_step = True
        while made_step:
            result = one_step_greedy(state, X, SigmaHat, k, pen_factor)
            state = result['State']
            made_step = result['madeStep']

        states.append(state) # Store generated state for each value of k

    best_idx = np.argmin([s.score for s in states]) # Choose best BIC from all states across k values
    final_state = states[best_idx] 

    return {
        'Adj': final_state.adj,
        'B': final_state.B,
        'variances': final_state.each_res_var,
    }
