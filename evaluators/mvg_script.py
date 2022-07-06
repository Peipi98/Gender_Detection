# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append('../')
from mlFunc import *
from validators import *
from classifiers import *
from classifiers import *
from validators import *
from prettytable import PrettyTable

def compute_MVG_score(Dte, D, L, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels):
    _, _, llrs = MVG(Dte, D, L)
    _, _, llrsn = naive_MVG(Dte, D, L)
    _, _, llrst = tied_cov_GC(Dte, D, L)
    _, _, llrsnt = tied_cov_naive_GC(Dte, D, L)
    
    MVG_res.append(llrs)
    MVG_naive.append(llrsn)
    MVG_t.append(llrst)
    MVG_nt.append(llrsnt)
    #MVG_labels.append(Lte)
    # MVG_labels = np.append(MVG_labels, Lte, axis=0)
    # MVG_labels = np.hstack(MVG_labels)
    return MVG_res, MVG_naive, MVG_t, MVG_nt

def evaluation(title, pi, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels):
    MVG_res = np.hstack(MVG_res)
    MVG_naive = np.hstack(MVG_naive)
    MVG_t = np.hstack(MVG_t)
    MVG_nt = np.hstack(MVG_nt)
    
    llrs_tot = compute_min_DCF(MVG_res, MVG_labels, pi, 1, 1)
    llrsn_tot = compute_min_DCF(MVG_naive, MVG_labels, pi, 1, 1)
    llrst_tot = compute_min_DCF(MVG_t, MVG_labels, pi, 1, 1)
    llrsnt_tot = compute_min_DCF(MVG_nt, MVG_labels, pi, 1, 1)
    
    
   
    plot_ROC(MVG_res, MVG_labels, 'MVG')
    plot_ROC(MVG_naive, MVG_labels, 'MVG + Naive')
    plot_ROC(MVG_t, MVG_labels, 'MVG + Tied')
    plot_ROC(MVG_nt, MVG_labels, 'MVG + Naive + Tied')


    #Cfn and Ctp are set to 1
    bayes_error_min_act_plot(MVG_res, MVG_labels, 'MVG', 0.4)
    bayes_error_min_act_plot(MVG_naive, MVG_labels, 'MVG + Naive', 1)
    bayes_error_min_act_plot(MVG_t, MVG_labels, 'MVG + Tied', 0.4)
    bayes_error_min_act_plot(MVG_nt, MVG_labels, 'MVG + Naive + Tied', 1)
    
    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["MVG", round(llrs_tot, 3)])
    t.add_row(["MVG naive", round(llrsn_tot,3)])
    t.add_row(["MVG tied", round(llrst_tot, 3)])
    t.add_row(["MVG naive + tied", round(llrsnt_tot, 3)])
    print(t)

if __name__ == '__main__':
    DTR, LTR = load("../Train.txt")
    DTE, LTE = load("../Test.txt")
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)
    
    MVG_res = []
    MVG_naive = []
    MVG_t = []
    MVG_nt = []
    MVG_labels = []
    
    PCA_mvg = []
    PCA_mvg_naive = []
    PCA_mvg_t = []
    PCA_mvg_nt = []
    
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
        
        MVG_labels = np.append(MVG_labels, Lte, axis=0)
        MVG_labels = np.hstack(MVG_labels)
        
        # Once we have computed our folds, we can try different models
        # RAW DATA
        
        MVG_res, MVG_naive, MVG_t, MVG_nt = compute_MVG_score(
            Dte, 
            D, 
            L, 
            MVG_res, 
            MVG_naive, 
            MVG_t, 
            MVG_nt, 
            MVG_labels)
        P = PCA(D, L, m = 10)
        DTR_PCA = numpy.dot(P.T, D)
        DTE_PCA = numpy.dot(P.T, Dte)
        
        # PCA m=10
        
        PCA_mvg, PCA_mvg_naive, PCA_mvg_t, PCA_mvg_nt = compute_MVG_score(
            DTE_PCA, 
            DTR_PCA, 
            L, 
            PCA_mvg, 
            PCA_mvg_naive, 
            PCA_mvg_t, 
            PCA_mvg_nt, 
            MVG_labels)    
    
    # π = 0.5 (our application prior)
    evaluation("minDCF: π=0.5", 0.5, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels)
    
    ###############################
    
    # π = 0.1
    evaluation("minDCF: π=0.1", 0.1, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels)
    
    ###############################

    # π = 0.9
    evaluation("minDCF: π=0.1", 0.9, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels)
