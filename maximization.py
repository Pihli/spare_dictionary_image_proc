import numpy as np


def maximization(X, R, nlsp):
    print("maximization")
    d, n = X.shape
    R = np.repeat(R, nlsp, axis=0)
    k = R.shape[1]

    nk = R.sum(axis=0)
    w = nk / nk.sum()
    means = np.matmul(X, R) / nk

    Sigma = np.zeros([k, d, d], np.float32)
    sqrtR = np.sqrt(R)

    for i in range(k):
        Xo = X - means[:, i].reshape([-1, 1])
        Xo = Xo * sqrtR[:, i].reshape([1, -1])
        Sigma[i, :, :] = np.matmul(Xo, Xo.T) / nk[i]
        Sigma[i, :, :] = Sigma[i, :, :] + np.eye(d) * (1e-6)

    model = dict()
    model["dim"] = d
    model["nmodels"] = k
    model["mixweights"] = w
    model["means"] = means
    model["covs"] = Sigma
    return model
