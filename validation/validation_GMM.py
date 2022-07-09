# -*- coding: utf-8 -*-
import sys
import numpy as np
import scipy
sys.path.append('../')
from mlFunc import *
from validators import *
from classifiers import *
from prettytable import PrettyTable
from Classifiers.GMM import GMM

def kfold_GMM(DTR, LTR):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    GMM_labels = []
    gmm_tot = []
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
        
        #CLASS PRIORS: WE CONSIDER A BALANCED APPLICATION
        prior_0 = 0.5
        prior_1 = 0.5
        
        #GMM MODELS
        
        optimal_m = 10
        optimal_comp = 2
        optimal_cov = 'full'
        optimal_alpha = 0.1
        optimal_psi = 0.01
        P_tr = PCA(D, L, optimal_m)
        P_te = PCA(Dte, Lte, optimal_m)
        
        reduced_dtr = np.dot(P_tr.T, D)
        reduced_dte = np.dot(P_te.T, Dte)
        
        #gmm = GMM(reduced_dtr, L, reduced_dte, Lte, [prior_0, prior_1], iterations=int(numpy.log2(optimal_comp)), alpha=optimal_alpha, psi=optimal_psi, typeOfGmm=optimal_cov)
        gmm = GMM(D, L, Dte, Lte, [prior_0, prior_1], iterations=int(numpy.log2(optimal_comp)), alpha=optimal_alpha, psi=optimal_psi, typeOfGmm=optimal_cov)

        gmm.train()
        print("llrs", gmm.llrs)
        min_dcf = gmm.compute_min_dcf()[0]

    return min_dcf
        
        
if __name__ == "__main__":
    DTR, LTR = load("../Train.txt")
    # gmm = initialize_GMM(DTR,4)
    # print(gmm[0][0])
    # print(gmm[0][1].shape)
    # print(gmm[0][2].shape)
    # gmm = GMM_EM(DTR, gmm)
    min_dcf = kfold_GMM(DTR, LTR)
    
        
    
        