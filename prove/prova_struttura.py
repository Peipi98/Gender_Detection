# -*- coding: utf-8 -*-
import sys
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
    
    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(numpy.hstack(Dtr[i + 1:]))
            L.append(numpy.hstack(Ltr[i + 1:]))
        elif i == k - 1:
            D.append(numpy.hstack(Dtr[:i]))
            L.append(numpy.hstack(Ltr[:i]))
        else:
            D.append(numpy.hstack(Dtr[:i]))
            D.append(numpy.hstack(Dtr[i + 1:]))
            L.append(numpy.hstack(Ltr[:i]))
            L.append(numpy.hstack(Ltr[i + 1:]))

        D = numpy.hstack(D)
        L = numpy.hstack(L)
        
        Dte = Dtr[i]
        Lte = Ltr[i]
        
        # Once we have computed our folds, we can try different models
        _, _, llrs = MVG(DTE, D, L)
        _, _, llrsn = naive_MVG(Dte, D, L)
        _, _, llrst = tied_cov_GC(Dte, D, L)
        _, _, llrsnt = tied_cov_naive_GC(Dte, D, L)
        
        MVG_res.append(llrs)
        MVG_naive.append(llrsn)
        MVG_t.append(llrst)
        MVG_nt.append(llrsnt)
    
    
    minDCF_MVG = compute_min_DCF(MVG_res, labels, pi, Cfn, Cfp)
    
    plot_ROC(llrs, LTE, 'MVG')
    plot_ROC(llrsn, LTE, 'MVG + Naive')
    plot_ROC(llrst, LTE, 'MVG + Tied')
    plot_ROC(llrsnt, LTE, 'MVG + Naive + Tied')


    #Cfn and Ctp are set to 1
    bayes_error_min_act_plot(llrs, LTE, 'MVG', 0.4)
    bayes_error_min_act_plot(llrsn, LTE, 'MVG + Naive', 1)
    bayes_error_min_act_plot(llrst, LTE, 'MVG + Tied', 0.4)
    bayes_error_min_act_plot(llrsnt, LTE, 'MVG + Naive + Tied', 1)
        