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
    

    h = {}
    
    for i in range(2):
        mu, C = ML_GAU(DTR[:, LTR==i])
        h[i] = (mu, C)
        
    SJoint = numpy.zeros((2, DTE.shape[1]))
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    classPriors = [0.999, 0.001]
    
    for label in range(2):
        mu, C = h[label]
        
        SJoint[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, C).ravel()) * classPriors[label] 
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + numpy.log(classPriors[label])
        
    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)
    
    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)
    
    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    
    accuracy_1 = (LTE == LPred1).sum() /LTE.size
    error_1 = 1 - accuracy_1
    
    accuracy_2 = (LTE == LPred2).sum() /LTE.size
    error_2 = 1 - accuracy_2
    #PCA(D, L)