import numpy as np
from causallearn.search.PermutationBased.GRaSP import grasp

def run_grasp(X, seed=42):
    n, p = X.shape

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Runs GRaSP - modernization of PC that identifies MEC via edges and unshielded colliders
    graph_result = grasp(X, score_func='local_score_BIC_from_cov')
    cpdag_raw = graph_result.graph

    # Convert to easily usable format - 1 at i,j for directed edge, 2 at i,j, j,i for undirected edge
    # This does create an UT graph if variables in causal order, but easier this way for display in python
    cpdag = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(p):
            if i == j:
                continue

            edge_ij = cpdag_raw[i, j]
            edge_ji = cpdag_raw[j, i]

            if edge_ij == -1 and edge_ji == 1:
                cpdag[i, j] = 1
            elif edge_ij == -1 and edge_ji == -1:
                cpdag[i, j] = 2

    return {"cpdag": cpdag}
