import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
#import sys

#sys.path.append("./functions")
from mlFunc import *

def kfold_cross(func, DTR, LTR, k):
    accuracy = []
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)
    
    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(numpy.hstack(Dtr[i+1:]))
            L.append(numpy.hstack(Ltr[i+1:]))
        elif i == k-1:
            D.append(numpy.hstack(Dtr[:i]))
            L.append(numpy.hstack(Ltr[:i]))
        else:
            D.append(numpy.hstack(Dtr[:i]))
            D.append(numpy.hstack(Dtr[i+1:]))
            L.append(numpy.hstack(Ltr[:i]))
            L.append(numpy.hstack(Ltr[i+1:]))
            
        D = numpy.hstack(D)
        L = numpy.hstack(L)
        
        DTE = Dtr[i]
        LTE = Ltr[i]
        #print(str(DTE) + " " + str(i))
        _, lpred  = func(DTE, LTE, D, L)
        acc, _ = test(LTE, lpred)
        accuracy.append(acc)
        
    return numpy.mean(accuracy)

def leave_one_out(func, DTR, LTR):
    accuracy = []
    
    for i in range(DTR.shape[1]):
        D = []
        L = []
        D.append(DTR[:,:i])
        D.append(DTR[:,i+1:])
        D = numpy.hstack(D)
        
        L.append(LTR[:i])
        L.append(LTR[i+1:])
        L = numpy.hstack(L)
        
        DTE = DTR[:, i]
        LTE = LTR[i]
        #print(str(DTE) + " " + str(i))
        _, lpred  = func(mcol(DTE), mcol(LTE), D, L)
        acc, _ = test(LTE, lpred)
        accuracy.append(acc)
    return numpy.mean(accuracy)
        
if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    
    DTE, LTE = load("Test.txt")
    # plot_hist(D, L)

    # We're starting with Multivariate Gaussian Classifier
    _, LPred2 = MGC(DTE, LTE, DTR, LTR)
    _, LP2n = naive_MGC(DTE, LTE, DTR, LTR)
    _, LP2t = tied_cov_GC(DTE, LTE, DTR, LTR)
    _, LP2nt = tied_cov_naive_GC(DTE, LTE, DTR, LTR)
    # logMGC accuracy
    log_acc, log_err = test(LTE, LPred2)
    log_acc_n, log_err_n = test(LTE, LP2n)
    log_acc_t, log_err_t = test(LTE, LP2t)
    log_acc_nt, log_err_nt = test(LTE, LP2nt)
    
    # print(leave_one_out(MGC, DTR, LTR))
    # print(leave_one_out(naive_MGC, DTR, LTR))
    # print(leave_one_out(tied_cov_GC, DTR, LTR))
    # print(leave_one_out(tied_cov_naive_GC, DTR, LTR))
    kfold_cross(MGC, DTR, LTR, 10)
    
    # DA CHIEDERE
    # Notiamo che i risultati di leave-one-out sono rispettivamente
    # più bassi rispetto ai precedenti non naive, ma più alti dei naive.
    # 0.9753333333333334 
    # 0.7031666666666667
    # 0.9755
    # 0.7048333333333333
    
    # Notiamo che le features sono molto correlate tra loro,
    # quindi non possiamo fare l'assunzione di indipendenza di Naive Bayes

    # PCA(D, L)
