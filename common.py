import numpy as np


def expectation(X, model, nlsp):
    means = model.means
    covs = model.covs
    w = model.mixweights

    n = X.size[1] / nlsp
    k = means.shape[1]
    logRho = np.zeros([n, k])

    for i in range(k):
        TemplogRho = loggausspdf(X, means[:, i], covs[:, :, i])
        Temp = TemplogRho.reshape([nlsp, n])
        logRho[:, i] = Temp.sum(axis=0)
    logRho = logRho + np.log(w)
    T = logsumexp(logRho, 2)
    llh = sum(T) / n
    logR = logRho - T
    R = np.exp(logR)
    return R, llh


def loggausspdf(*args):
    pass


def logsumexp(*args):
    pass


def initialization(X, init, nlsp):
    index = np.arange(0, nlsp, X.shape[1])
    X = X[:, index]
    [d, n] = X.shape
    if type(init) is dict:  # initialize with a model
        R = expectation(X, init, None)
    elif len(init) == 1:  # random initialization
        k = init
        idx = np.random.choice(n, k, replace=False)
        m = X[:, idx]
        label = (np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape([-1, 1]) / 2).argmax(axis=0)
        u, label = np.unique(label, return_inverse=True)
        while k != len(u):
            idx = np.random.choice(n, k, replace=False)
            m = X[:, idx]
            label = np.argmax(np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape([-1, 1]) / 2, axis=0)
            u, label = np.unique(label, return_inverse=True)
        R = np.zeros([n, k])
        R[np.arange(n), label] = 1
    elif init.shape[0] == 1 and init.shape[1] == n:
        label = init
        k = max(label)
        R = np.zeros([n, k])
        R[np.arange(n), label] = 1
    elif init.shape[0] == d:
        k = init.shape[1]
        m = init
        label = np.argmax(np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape([-1, 1]) / 2, axis=0)
        R = np.zeros([n, k])
        R[np.arange(n), label] = 1
    else:
        raise ('ERROR: init is not valid.')
    return R


# [model,llh,label]= emgm(X, init,nlsp)
def emgm(X, init, nlsp):
    print("EM for PG-GMM: running ... ")
    R = initialization(X, init, nlsp)
