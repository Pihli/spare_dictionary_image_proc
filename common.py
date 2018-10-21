# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:03:24 2018

@author: Touch
"""
import numpy as np


#def expectation(X, model,nlsp):
#    means = model.means;
#    covs = model.covs;
#    w = model.mixweights;
#    
#    n = X.size[1]/nlsp;
#    k = size(means,2);
#    logRho = zeros(n,k);
#    
#    for i in range(k):
#        TemplogRho = loggausspdf(X,means[:,i],covs[:,:,i]);
#        Temp = reshape(TemplogRho,[nlsp n]);
#        logRho(:,i) = sum(Temp);
#    end
#    logRho = bsxfun(@plus,logRho,log(w));
#    T = logsumexp(logRho,2);
#    llh = sum(T)/n; % loglikelihood
#    logR = bsxfun(@minus,logRho,T);
#    R = exp(logR);
#    return R, llh


def initialization(X, init,nlsp):
    index = np.arange(0,nlsp,size(X,2))
    X = X[:,index]
    [d,n] = size(X);
    if type(init) is dict:  # initialize with a model
        R  = expectation(X,init)
    elif len(init) == 1:  # random initialization
        k = init
        idx = np.random.randint(n,size = [k,1])
        m = X[:,idx]
        label = (np.matmul(m.T,X)-(m*m).T.sum(axis=1)/2).argmax(axis=0)
        [u,~,label] = unique(label);
        while k ~= length(u)
            idx = randsample(n,k);
            m = X(:,idx);
            [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
            [u,~,label] = unique(label);
        end
        R = full(sparse(1:n,label,1,n,k,n));
    elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
        label = init;
        k = max(label);
        R = full(sparse(1:n,label,1,n,k,n));
    elseif size(init,1) == d  %initialize with only centers
        k = size(init,2);
        m = init;
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
        R = full(sparse(1:n,label,1,n,k,n));
    else
        error('ERROR: init is not valid.');
    end
    return R






#[model,llh,label]= emgm(X, init,nlsp)
def emgm(X, init,nlsp):
    print("EM for PG-GMM: running ... ")
    R = intialization(X,init,nlsp)
    