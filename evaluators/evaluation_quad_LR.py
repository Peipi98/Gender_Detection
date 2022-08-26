# -*- coding: utf-8 -*-
import sys
from os import path
import numpy as np

from validation.validation_quad_LR import kfold_QUAD_LR_tuning

sys.path.append('../')
from validators import *
from prettytable import PrettyTable

def compare_QLR_val_eval(x, D, title, base=10):
    plt.figure()
    plt.plot(x, D[0][0], color='r', label='minDCF(π=0.5)[Eval]')
    plt.plot(x, D[0][2], color='b', label='minDCF(π=0.1)[Eval]')
    plt.plot(x, D[0][1], color='g', label='minDCF(π=0.9)[Eval]')
    plt.plot(x, D[1][0], color='r', linestyle='dashed', label='minDCF(π=0.5)[Val]')
    plt.plot(x, D[1][2], color='b', linestyle='dashed', label='minDCF(π=0.1)[Val]')
    plt.plot(x, D[1][1], color='g', linestyle='dashed', label='minDCF(π=0.9)[Val]')
    plt.xscale("log", base=base)
    plt.xlim([min(x), max(x)])
    #plt.ylim(0, ylim)
    plt.legend()
    plt.savefig('./images/QuadLogReg_eval_' + title + '.svg')
    plt.show()

def validate_LR(scores, LR_labels, appendToTitle, l, pi):
    scores_tot_05 = compute_min_DCF(scores, LR_labels, 0.5, 1, 1)
    scores_tot_01 = compute_min_DCF(scores, LR_labels, 0.1, 1, 1)
    scores_tot_09 = compute_min_DCF(scores, LR_labels, 0.9, 1, 1)
    # plot_ROC(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l), 0.4)

    t = PrettyTable(["Type", "π=0.5", "π=0.1", "π=0.9"])
    t.title = appendToTitle
    t.add_row(['QUAD_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    print(t)


def evaluate_LR(DTR, LTR, DTE, LTE, l, pi, appendToTitle, PCA_Flag=True, zscore_Flag=False, gauss_Flag=False):

    if zscore_Flag is True:
        DTR, DTE = znorm(DTR, DTE)

    if gauss_Flag is True:
        D_training = DTR
        DTR = gaussianize_features(DTR, DTR)
        DTE = gaussianize_features(D_training, DTE)

    def vecxxT(x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        return xxT

    expanded_DTR = numpy.apply_along_axis(vecxxT, 0, DTR)
    expanded_DTE = numpy.apply_along_axis(vecxxT, 0, DTE)
    phi = numpy.vstack([expanded_DTR, DTR])

    phi_DTE = numpy.vstack([expanded_DTE, DTE])

    scores = quad_logistic_reg_score(phi, LTR, phi_DTE, l, pi=pi)
    validate_LR(scores, LTE, appendToTitle, l, pi=pi)

    # PCA m=10
    P = PCA(DTR, LTR, m=10)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)

    PCA_LR_scores = quad_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, pi=pi)
    validate_LR(PCA_LR_scores, LTE, appendToTitle + 'PCA_m10_', l, pi=pi)



def quadlr_tuning(DTR, LTR, DTE, LTE, xi, zscore_Flag=False, gauss_Flag=False):

    if zscore_Flag is True:
        DTR, DTE = znorm(DTR, DTE)

    if gauss_Flag is True:
        D_training = DTR
        DTR = gaussianize_features(DTR, DTR)
        DTE = gaussianize_features(D_training, DTE)

    def vecxxT(x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        return xxT

    expanded_DTR = numpy.apply_along_axis(vecxxT, 0, DTR)
    expanded_DTE = numpy.apply_along_axis(vecxxT, 0, DTE)
    phi = numpy.vstack([expanded_DTR, DTR])

    phi_DTE = numpy.vstack([expanded_DTE, DTE])

    scores = quad_logistic_reg_score(phi, LTR, phi_DTE, xi)

    return scores, LTE


def evaluation_quad_LR(DTR, LTR, DTE, LTE, L, appendToTitle, gauss_Flag=False, zscore_Flag=False):
    for l in L:  # l is a constant, not an array
        evaluate_LR(DTR, LTR, DTE, LTE, l, 0.5, appendToTitle, gauss_Flag=gauss_Flag, zscore_Flag=zscore_Flag)
        evaluate_LR(DTR, LTR, DTE, LTE, l, 0.1, appendToTitle, gauss_Flag=gauss_Flag, zscore_Flag=zscore_Flag)
        evaluate_LR(DTR, LTR, DTE, LTE, l, 0.9, appendToTitle, gauss_Flag=gauss_Flag, zscore_Flag=zscore_Flag)
    x = numpy.logspace(-5, 1, 30)
    y = numpy.array([])
    eval_05 = numpy.array([])
    eval_09 = numpy.array([])
    eval_01 = numpy.array([])

    val = numpy.array([])
    val_05 = numpy.array([])
    val_09 = numpy.array([])
    val_01 = numpy.array([])
    filepath = './evaluators/evaluation_QLR_' + appendToTitle + '.npz'
    if path.exists(filepath):
        arrays = np.load(filepath)
        y = arrays['eval']
        val = arrays['val']
    else:

        for xi in x:
            scores, labels = quadlr_tuning(DTR, LTR, DTE, LTE, xi, zscore_Flag, gauss_Flag)
            eval_05 = numpy.hstack((eval_05, bayes_error_plot_compare(0.5, scores, labels)))
            eval_09 = numpy.hstack((eval_09, bayes_error_plot_compare(0.9, scores, labels)))
            eval_01 = numpy.hstack((eval_01, bayes_error_plot_compare(0.1, scores, labels)))

            scores, labels = kfold_QUAD_LR_tuning(DTR, LTR, xi, zscore_Flag=zscore_Flag, gauss_Flag=gauss_Flag)
            val_05 = numpy.hstack((val_05, bayes_error_plot_compare(0.5, scores, labels)))
            val_09 = numpy.hstack((val_09, bayes_error_plot_compare(0.9, scores, labels)))
            val_01 = numpy.hstack((val_01, bayes_error_plot_compare(0.1, scores, labels)))

        y = numpy.hstack((y, eval_05))
        y = numpy.vstack((y, eval_09))
        y = numpy.vstack((y, eval_01))

        val = numpy.hstack((val, val_05))
        val = numpy.vstack((val, val_09))
        val = numpy.vstack((val, val_01))

        np.savez('./evaluators/evaluation_QLR_' + appendToTitle + '.npz', val=val, eval=y)

    compare_QLR_val_eval(x, [y, val], appendToTitle)

