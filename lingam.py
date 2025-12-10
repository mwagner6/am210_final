import numpy as np
from causallearn.search.FCMBased import lingam

def run_lingam(X, output=False):
    model = lingam.ICALiNGAM()
    model.fit(X)

    adj_weighted = model.adjacency_matrix_.T
    threshold = 0.05 * np.max(adj_weighted)
    # For real-world data - accept edges for some threshold of relevance (don't want a ton of noisy edges)
    adj_binary = (np.abs(adj_weighted) > threshold).astype(int)

    # Estimate error variances from residuals
    p = X.shape[1]
    variances = np.zeros(p)

    # Compute variances by subtracting off causal effect from parents from true variance
    for j in range(p):
        parents = np.where(adj_binary[:, j] == 1)[0]
        if len(parents) == 0:
            variances[j] = np.var(X[:, j], ddof=1)
        else:
            predicted = X[:, parents] @ adj_weighted[j, parents]
            residuals = X[:, j] - predicted
            variances[j] = np.var(residuals, ddof=1)

    return {
        'Adj': adj_binary,
        'B': adj_weighted,
        'variances': variances
    }
