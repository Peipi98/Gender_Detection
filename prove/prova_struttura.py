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
        
        # Once we have computed our folds, we can try different models
        _, _, llrs = MVG(Dte, D, L)
        _, _, llrsn = naive_MVG(Dte, D, L)
        _, _, llrst = tied_cov_GC(Dte, D, L)
        _, _, llrsnt = tied_cov_naive_GC(Dte, D, L)
        
        MVG_res.append(llrs)
        MVG_naive.append(llrsn)
        MVG_t.append(llrst)
        MVG_nt.append(llrsnt)
        #MVG_labels.append(Lte)
        MVG_labels = np.append(MVG_labels, Lte, axis=0)
        MVG_labels = np.hstack(MVG_labels)
    
    MVG_res = np.hstack(MVG_res)
    MVG_naive = np.hstack(MVG_naive)
    MVG_t = np.hstack(MVG_t)
    MVG_nt = np.hstack(MVG_nt)
    
    llrs_tot = compute_min_DCF(MVG_res, MVG_labels, 0.5, 1, 1)
    llrsn_tot = compute_min_DCF(MVG_naive, MVG_labels, 0.5, 1, 1)
    llrst_tot = compute_min_DCF(MVG_t, MVG_labels, 0.5, 1, 1)
    llrsnt_tot = compute_min_DCF(MVG_nt, MVG_labels, 0.5, 1, 1)
   
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
    t.title = "minDCF: π=0.5"
    t.add_row(["MVG", llrs_tot])
    t.add_row(["MVG naive", llrsn_tot])
    t.add_row(["MVG tied", llrst_tot])
    t.add_row(["MVG naive + tied", llrsnt_tot])
    print(t)
    
    ###############################
    
    # π = 0.1
    llrs_tot = compute_min_DCF(MVG_res, MVG_labels, 0.1, 1, 1)
    llrsn_tot = compute_min_DCF(MVG_naive, MVG_labels, 0.1, 1, 1)
    llrst_tot = compute_min_DCF(MVG_t, MVG_labels, 0.1, 1, 1)
    llrsnt_tot = compute_min_DCF(MVG_nt, MVG_labels, 0.1, 1, 1)
    
    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.1"
    t.add_row(["MVG", llrs_tot])
    t.add_row(["MVG naive", llrsn_tot])
    t.add_row(["MVG tied", llrst_tot])
    t.add_row(["MVG naive + tied", llrsnt_tot])
    print(t)
    
    ###############################

    # π = 0.9
    llrs_tot = compute_min_DCF(MVG_res, MVG_labels, 0.9, 1, 1)
    llrsn_tot = compute_min_DCF(MVG_naive, MVG_labels, 0.9, 1, 1)
    llrst_tot = compute_min_DCF(MVG_t, MVG_labels, 0.9, 1, 1)
    llrsnt_tot = compute_min_DCF(MVG_nt, MVG_labels, 0.9, 1, 1)
    
    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.9"
    t.add_row(["MVG", llrs_tot])
    t.add_row(["MVG naive", llrsn_tot])
    t.add_row(["MVG tied", llrst_tot])
    t.add_row(["MVG naive + tied", llrsnt_tot])
    print(t)