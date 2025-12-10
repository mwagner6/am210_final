from scipy import stats

# Returns 0: all non-gaussian. Returns 1: all gaussian. Returns 2: mixed
def test_gaussianity(X, alpha=0.05):
    p = X.shape[1]

    is_gaussian = []

    for j in range(p):
        _, jb_p = stats.jarque_bera(X[:, j])
        is_gaussian.append(jb_p > alpha)

    if not any(is_gaussian):
        return 0
    elif all(is_gaussian):
        return 1
    else:
        return 2
