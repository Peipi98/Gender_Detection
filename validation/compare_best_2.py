#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pylab

from classifiers import weighted_logistic_reg_score
from evaluators.evaluation_GMM import evaluation_GMM_ncomp
from validation_GMM import kfold_GMM, validation_GMM_ncomp
from validators import bayes_error_plot, confusion_matrix_binary


def bayes_error_plot_2best(D, L, pi, title, ylim):
    p = np.linspace(-3, 3, 21)
    pylab.title(title)
    pylab.plot(p, bayes_error_plot(p, D[0], L, minCost=False), color='r', label='GMM_RAW_actDCF')
    pylab.plot(p, bayes_error_plot(p, D[0], L, minCost=True), 'r--', label='GMM_RAW_minDCF')
    #pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False, th=-np.log(pi / (1-pi))), color='y')
    
    pylab.plot(p, bayes_error_plot(p, D[1], L, minCost=False), color='b', label='WLR_RAW_actDCF')
    pylab.plot(p, bayes_error_plot(p, D[1], L, minCost=True), 'b--', label='WLR_RAW_minDCF')
    #pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False, th=-np.log(pi / (1-pi))), color='y')
    
    pylab.ylim(0, ylim)
    pylab.savefig('../images/comparison/DCF_2best' + title + '.png')
    pylab.show()

def ROC_2best(D, L, pi, title):
    thresholds = np.array(D[0])
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(D[0] > t)
        conf = confusion_matrix_binary(Pred, L)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    pylab.plot(FPR, TPR)

    thresholds = np.array(D[1])
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(D[1] > t)
        conf = confusion_matrix_binary(Pred, L)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    pylab.plot(FPR, TPR)

    pylab.title(title)
    pylab.savefig('../images/comparison/ROC_2best' + title + '.png')
    pylab.show()
    
def compute_2best_plots(DTR, LTR, DTE, LTE):
    # GMM_llrst_raw, GMM_labels_raw = validation_GMM_ncomp(DTR, LTR, 0.5, 2)
    # WLR_scores = weighted_logistic_reg_score(DTR, LTR, DTE, 1e-4)
    GMM_llrst_raw = evaluation_GMM_ncomp('', DTR, LTR, DTE, LTE, 0.5, 2)
    WLR_scores = weighted_logistic_reg_score(DTR, LTR, DTE, 1e-4)
    bayes_error_plot_2best([GMM_llrst_raw, WLR_scores], LTE ,0.5, '', 0.4)
    ROC_2best([GMM_llrst_raw, WLR_scores],LTE,0.5,'')
    # Put here models to be compared
    
    #bayes_error_plot_2best([D1, D2], [L1,L2], 0.5, "", 0.4)