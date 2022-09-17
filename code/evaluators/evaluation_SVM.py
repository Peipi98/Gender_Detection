# -*- coding: utf-8 -*-
import sys

import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def evaluate_SVM(DTR, LTR, DTE, LTE, K, C, appendToTitle, PCA_Flag=True):
    scores_append = []
    SVM_labels = []

    wStar, _ = train_SVM_linear(DTR, LTR, C=C, K=K)

    DTEEXT = numpy.vstack([DTE, K * numpy.ones((1, DTE.shape[1]))])

    scores = numpy.dot(wStar.T, DTEEXT).ravel()
    scores_append.append(scores)

    SVM_labels = np.append(SVM_labels, LTE, axis=0)
    SVM_labels = np.hstack(SVM_labels)

    scores_append = np.hstack(scores_append)
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.5, 1, 1)

    # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.5"
    t.add_row(['SVM, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.1, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.1"
    t.add_row(['SVM, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.9"
    t.add_row(['SVM, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)


def svm_tuning(DTR, LTR,DTE, LTE, K, C):
    scores_append = []
    labels = []

    wStar, _ = train_SVM_linear(DTR, LTR, C=C, K=K)
    DTEEXT = numpy.vstack([DTE, K * numpy.ones((1, DTE.shape[1]))])

    scores = numpy.dot(wStar.T, DTEEXT).ravel()
    scores_append.append(scores)

    labels = np.append(labels, LTE, axis=0)
    labels = np.hstack(labels)

    return np.hstack(scores_append), labels


def evaluation_SVM(DTR, LTR, DTE, LTE, K_arr, C_arr, appendToTitle, PCA_Flag=True):
    for K in K_arr:
        for C in C_arr:
            evaluate_SVM(DTR, LTR, DTE, LTE, K, C, appendToTitle, PCA_Flag=False)
    x = numpy.logspace(-3, 2, 14)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    for xi in x:
        scores, labels = svm_tuning(DTR, LTR, DTE, LTE, 1.0, xi)
        y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
        y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
        y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    plot_DCF(x, y, 'C', appendToTitle + 'Linear_SVM_minDCF_comparison')
