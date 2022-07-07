# -*- coding: utf-8 -*-
import sys
import numpy as np
import scipy
sys.path.append('../')
from mlFunc import *
from validators import *
from classifiers import *
from classifiers import *
from validators import *
from prettytable import PrettyTable

def initialize_GMM(D, n):
    gmm = []
    
    for i in range(n):
        weights = 1/n
        mu = mcol(np.array(D.mean(1)))
        C = np.matrix(empirical_covariance(D, mu))
        
        gmm.append((weights, mu, C))
    
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]
    
    

def GMM_EM(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G,N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma=P[g, :]
            Z = gamma.sum()
            F=(mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu=mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        #print(llNew)
    print(llNew-llOld)
    return gmm

def kfold_GMM(DTR, LTR):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    GMM_labels = []
    gmm_tot = []
    for n in range(2, 5):
        for i in range(k):
            D = []
            L = []
            if i == 0:
                D.append(np.hstack(Dtr[i + 1:]))
                L.append(np.hstack(Ltr[i + 1:]))
            elif i == k - 1:
                D.append(np.hstack(Dtr[:i]))
                L.append(np.hstack(Ltr[:i]))
            else:
                D.append(np.hstack(Dtr[:i]))
                D.append(np.hstack(Dtr[i + 1:]))
                L.append(np.hstack(Ltr[:i]))
                L.append(np.hstack(Ltr[i + 1:]))
    
            D = np.hstack(D)
            L = np.hstack(L)
    
            Dte = Dtr[i]
            Lte = Ltr[i]
            
            gmm = initialize_GMM(D, n)
            
            gmm = GMM_EM(D, gmm)
            
            gmm_tot.append(gmm)
    return gmm_tot
        
        
if __name__ == "__main__":
    DTR, LTR = load("../Train.txt")
    # gmm = initialize_GMM(DTR,4)
    # print(gmm[0][0])
    # print(gmm[0][1].shape)
    # print(gmm[0][2].shape)
    # gmm = GMM_EM(DTR, gmm)
    gmm_tot = kfold_GMM(DTR, LTR)
    
        
    
        