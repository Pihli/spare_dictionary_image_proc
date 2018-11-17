import numpy as np
from expectation import expectation
from maximization import maximization


def initialization(X, init, nlsp):
    print("initialization")
    index = np.arange(0, X.shape[1], nlsp)
    X = X[:, index]
    [d, n] = X.shape
    if type(init) is dict:  # initialize with a model
        R = expectation(X, init, None)
    elif init.size == 1:  # random initialization
        k = init
        idx = np.random.choice(n, k, replace=False)
        m = X[:, idx]
        label = (np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape([-1, 1]) / 2).argmax(axis=0)
        u, label = np.unique(label, return_inverse=True)
        while k != u.size:
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
    label = R.argmax(axis=1)
    R = R[:, np.unique(label)]

    tol = 1e-10
    maxiter = 100
    llh = -np.inf * np.ones(maxiter, dtype=np.float16)
    converged = False
    t = 0
    while not converged and t < maxiter:
        t += 1
        model = maximization(X, R, nlsp)
        R, llh[t] = expectation(X, model, nlsp)
        print('Iteration %d of %d, logL: %.2f' % (t, maxiter, llh[t]))
        # subplot(1, 2, 1)
        # plot(llh(1: t), 'o-')
        label = R.argmax(axis=1)
        u = np.unique(label)
        if R.shape[1] != u.size:
            R = R[:, u]
        # remove empty components
        else:
            converged = llh[t] - llh[t - 1] < tol * abs(llh[t])

    model["k"] = R.shape[1]
    if converged:
        print('Converged in %d steps.' % t - 1)
    else:
        print('Not converged in %d steps.' % maxiter)
    return model, llh, label
