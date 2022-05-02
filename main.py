import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
import sys
sys.path.append("./functions")
from mlFunc import *

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    #plot_hist(D, L)
    
    
    #We're starting with Multivariate Gaussian Classifier
    _, LPred2 = MGC(DTE, LTE, DTR, LTR)
    _, LP2n = naive_MGC(DTE, LTE, DTR, LTR)
    _, LP2t = tied_cov_GC(DTE, LTE, DTR, LTR)
    _, LP2nt = tied_cov_naive_GC(DTE, LTE, DTR, LTR)
    # logMGC accuracy
    log_acc, log_err = test(LTE, LPred2)
    log_acc_n, log_err_n = test(LTE, LP2n)
    log_acc_t, log_err_t = test(LTE, LP2t)
    log_acc_nt, log_err_nt = test(LTE, LP2nt)
    
    # Notiamo che le features sono molto correlate tra loro,
    # quindi non possiamo fare l'assunzione di indipendenza di Naive Bayes
    
    #PCA(D, L)