import numpy as np


def expectation(X, model, nlsp):
    print("expectation")
    means = model["means"]
    covs = model["covs"]
    w = model["mixweights"]
    n = X.shape[1] // nlsp
    k = means.shape[1]
    logRho = np.zeros([n, k], np.float32)
    for i in range(k):
        TemplogRho = loggausspdf(X, means[:, i].reshape([-1, 1]), covs[i, :, :])
        Temp = TemplogRho.reshape([n, nlsp])
        logRho[:, i] = Temp.sum(axis=1)
    logRho = logRho + np.log(w)
    T = logsumexp(logRho, 1)
    llh = T.mean()
    logR = logRho - T
    R = np.exp(logR)
    return R, llh


def loggausspdf(x, mu, sigma):
    d = x.shape[0]
    try:
        u = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        print(sigma)
        raise
    # matlab will return u, p
    Q = np.linalg.solve(u, x - mu)
    q = np.sum(Q * Q, axis=0)
    c = d * np.log(2 * np.pi) + np.log(sigma.diagonal()).sum()
    y = -(c + q) / 2
    return y


def logsumexp(x, dim=0):
    x = np.asmatrix(x)
    y = x.max(axis=dim)
    s = y + np.log(np.exp(x - y).sum(axis=dim))
    ind = np.where(~np.isfinite(s))[0]
    if ind.size:
        s[ind] = y[ind]
    return s.A
