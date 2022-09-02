# -*- coding: utf-8 -*-
import sys

import numpy as np

from validation.validation_weighted_LR import kfold_WEIGHTED_LR_tuning

sys.path.append('../')
from validators import *
from prettytable import PrettyTable
import pylab
import matplotlib.pyplot as plt

def compare_LR_val_eval(x, D, title, base=10):
    plt.figure()
    plt.plot(x, D[0][0], color='r', label='minDCF(π=0.5)[Eval]')
    plt.plot(x, D[0][2], color='b', label='minDCF(π=0.1)[Eval]')
    plt.plot(x, D[0][1], color='g', label='minDCF(π=0.9)[Eval]')
    plt.plot(x, D[1][0], color='r', linestyle='dashed', label='minDCF(π=0.5)[Val]')
    plt.plot(x, D[1][2], color='b', linestyle='dashed', label='minDCF(π=0.1)[Val]')
    plt.plot(x, D[1][1], color='g', linestyle='dashed', label='minDCF(π=0.9)[Val]')
    plt.xscale("log", base=base)
    plt.xlim([min(x), max(x)])
    plt.legend()
    plt.savefig('./images/LogReg_eval_' + title + '.svg')
    plt.show()

def validate_LR(scores, LR_labels, appendToTitle, l, pi):
    scores_tot_05 = compute_min_DCF(scores, LR_labels, 0.5, 1, 1)
    scores_tot_01 = compute_min_DCF(scores, LR_labels, 0.1, 1, 1)
    scores_tot_09 = compute_min_DCF(scores, LR_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "π=0.5", "π=0.1", "π=0.9"])
    t.title = appendToTitle
    t.add_row(['WEIGHTED_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    print(t)


def evaluate_LR(DTR, LTR, DTE, LTE, l, pi, appendToTitle, PCA_Flag=True, zscore_Flag=False, gauss_Flag=False):

    if zscore_Flag is True:
        DTR, DTE = znorm(DTR, DTE)

    if gauss_Flag is True:
        D_training = DTR
        DTR = gaussianize_features(DTR, DTR)
        DTE = gaussianize_features(D_training, DTE)

    scores = weighted_logistic_reg_score(DTR, LTR, DTE, l, pi)
    validate_LR(scores, LTE, appendToTitle, l, pi)

    # PCA m=10
    P = PCA(DTR, LTR, m=10)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)

    PCA_LR_scores = weighted_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, pi=pi)
    validate_LR(PCA_LR_scores, LTE, appendToTitle + 'PCA_m10_', l, pi=pi)


def lr_tuning(DTR, LTR, DTE, LTE, xi, zscore_Flag=False, gauss_Flag=False):

    if zscore_Flag is True:
        DTR, DTE = znorm(DTR, DTE)

    if gauss_Flag is True:
        D_training = DTR
        DTR = gaussianize_features(DTR, DTR)
        DTE = gaussianize_features(D_training, DTE)

    scores = weighted_logistic_reg_score(DTR, LTR, DTE, xi)

    return scores, LTE


def evaluation_weighted_LR(DTR, LTR, DTE, LTE, L, appendToTitle, gauss_Flag=False, zscore_Flag=False):

    for l in L:  # l is a constant, not an array
        evaluate_LR(DTR, LTR, DTE, LTE, l, 0.5, appendToTitle, gauss_Flag=gauss_Flag, zscore_Flag=zscore_Flag)
        evaluate_LR(DTR, LTR, DTE, LTE, l, 0.1, appendToTitle, gauss_Flag=gauss_Flag, zscore_Flag=zscore_Flag)
        evaluate_LR(DTR, LTR, DTE, LTE, l, 0.9, appendToTitle, gauss_Flag=gauss_Flag, zscore_Flag=zscore_Flag)

    x = numpy.logspace(-5, 1, 30)
    y = numpy.array([])
    val = numpy.array([])
    eval_05 = numpy.array([])
    eval_09 = numpy.array([])
    eval_01 = numpy.array([])

    val_05 = numpy.array([])
    val_09 = numpy.array([])
    val_01 = numpy.array([])

    for xi in x:
        scores, labels_val = lr_tuning(DTR, LTR, DTE, LTE, xi, zscore_Flag, gauss_Flag)
        eval_05 = numpy.hstack((eval_05, bayes_error_plot_compare(0.5, scores, labels_val)))
        eval_09 = numpy.hstack((eval_09, bayes_error_plot_compare(0.9, scores, labels_val)))
        eval_01 = numpy.hstack((eval_01, bayes_error_plot_compare(0.1, scores, labels_val)))

        scores, labels = kfold_WEIGHTED_LR_tuning(DTR, LTR, xi, zscore_Flag=zscore_Flag, gauss_Flag=gauss_Flag)
        val_05 = numpy.hstack((val_05, bayes_error_plot_compare(0.5, scores, labels)))
        val_09 = numpy.hstack((val_09, bayes_error_plot_compare(0.9, scores, labels)))
        val_01 = numpy.hstack((val_01, bayes_error_plot_compare(0.1, scores, labels)))
    y = numpy.hstack((y, eval_05))
    y = numpy.vstack((y, eval_09))
    y = numpy.vstack((y, eval_01))

    val = numpy.hstack((val, val_05))
    val = numpy.vstack((val, val_09))
    val = numpy.vstack((val, val_01))

    print("y shape: ", y.shape)
    print("val shape: ", val.shape)

    compare_LR_val_eval(x, [y, val], appendToTitle)
    # plot_DCF(x, y, 'lambda', appendToTitle + 'EVAL_WLR_minDCF_comparison')
