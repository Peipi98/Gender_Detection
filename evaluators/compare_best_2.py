#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pylab
from mlFunc import mrow
from classifiers import weighted_logistic_reg_score, tied_cov_GC
from evaluators.evaluation_GMM import evaluation_GMM_ncomp
from mlFunc import train_SVM_linear, train_SVM_RBF
from validators import bayes_error_plot, confusion_matrix_binary


def bayes_error_plot_2best(D, L, pi, title, ylim):
    p = np.linspace(-3, 3, 21)
    pylab.title(title)
    pylab.plot(p, bayes_error_plot(p, D[0], L, minCost=False), color='r', label='GMM_RAW_actDCF')
    pylab.plot(p, bayes_error_plot(p, D[0], L, minCost=True), 'r--', label='GMM_RAW_minDCF')

    pylab.plot(p, bayes_error_plot(p, D[1], L, minCost=False), color='b', label='MVG_TIED_RAW_actDCF')
    pylab.plot(p, bayes_error_plot(p, D[1], L, minCost=True), 'b--', label='MVG_TIED_minDCF')

    pylab.ylim(0, ylim)
    pylab.legend()
    #pylab.savefig('../images/comparison/DCF_2best' + title + '.png')
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
    pylab.plot(FPR, TPR, label="GMM")

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
    pylab.plot(FPR, TPR, label="MVG_TIED")

    pylab.title(title)
    pylab.legend()
    #pylab.savefig('../images/comparison/ROC_2best' + title + '.png')
    pylab.show()
    
def compute_2best_plots(DTR, LTR, DTE, LTE):
    # GMM_llrst_raw, GMM_labels_raw = validation_GMM_ncomp(DTR, LTR, 0.5, 2)
    # WLR_scores = weighted_logistic_reg_score(DTR, LTR, DTE, 1e-4)


    GMM_llrst_raw = evaluation_GMM_ncomp('', DTR, LTR, DTE, LTE, 0.5, 2)
    #WLR_scores = weighted_logistic_reg_score(DTR, LTR, DTE, 1e-4)

    C = 1.0
    K = 1.0
    # wStar, primal, dual, gap = train_SVM_linear(DTR, LTR, C=C, K=K)
    # DTEEXT = np.vstack([DTE, K * np.ones((1, DTE.shape[1]))])
    # second_scores = np.dot(wStar.T, DTEEXT).ravel()

    # Z = np.zeros(LTR.shape)
    # Z[LTR == 1] = 1
    # Z[LTR == 0] = -1
    # gamma = 0.001
    # aStar, loss = train_SVM_RBF(DTR, LTR, C=1.0, K=K, gamma=gamma)
    #
    # kern = np.zeros((DTR.shape[1], DTE.shape[1]))
    # for i in range(DTR.shape[1]):
    #     for j in range(DTE.shape[1]):
    #         kern[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K
    #
    # second_scores = np.sum(np.dot(aStar * mrow(Z), kern), axis=0)
    #
    # bayes_error_plot_2best([GMM_llrst_raw, second_scores], LTE, 0.5, '', 0.4)
    # ROC_2best([GMM_llrst_raw, second_scores], LTE, 0.5, '')

    _, _, second_scores = tied_cov_GC(DTE, DTR, LTR)
    bayes_error_plot_2best([GMM_llrst_raw, second_scores], LTE, 0.5, '', 0.4)
    ROC_2best([GMM_llrst_raw, second_scores], LTE, 0.5, '')

    # Put here models to be compared
    
    #bayes_error_plot_2best([D1, D2], [L1,L2], 0.5, "", 0.4)