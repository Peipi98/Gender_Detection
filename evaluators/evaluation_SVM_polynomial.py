# -*- coding: utf-8 -*-
import sys

import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def evaluate_SVM_polynomial(DTR, LTR, DTE, LTE, K, costant, appendToTitle, C=1.0, degree=2, PCA_Flag=True):
    scores_append = []
    SVM_labels = []

    aStar, loss = train_SVM_polynomial(DTR, LTR, C=C, constant=costant, degree=degree, K=K)
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    kernel = (numpy.dot(DTR.T, DTE) + costant) ** degree + K * K
    scores = numpy.sum(numpy.dot(aStar * mrow(Z), kernel), axis=0)
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
    t.add_row(['SVM_POLYNOMIAL, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.1, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.1"
    t.add_row(['SVM_POLYNOMIAL, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.9"
    t.add_row(['SVM_POLYNOMIAL, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)


def kfold_SVM_calibration(DTR, LTR, K, C):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    LR_labels = []

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
        print(i)
        wStar, primal, dual, gap = train_SVM_linear(D, L, C=C, K=K)
        DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])

        scores = numpy.dot(wStar.T, DTEEXT).ravel()
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), LR_labels


def evaluation_SVM_polynomial(DTR, LTR, DTE, LTE, K_arr, C, appendToTitle, CON_array, PCA_Flag=True):
    for costant in CON_array:
        for degree in [2]:
            for K in K_arr:
                evaluate_SVM_polynomial(DTR, LTR, DTE, LTE, K, costant, appendToTitle, C=C, degree=degree, PCA_Flag=False)

    # x = numpy.logspace(-3, 2, 15)
    # y = numpy.array([])
    # y_05 = numpy.array([])
    # y_09 = numpy.array([])
    # y_01 = numpy.array([])
    # for xi in x:
    #     scores, labels = kfold_SVM_calibration(DTR, LTR, 1.0, xi)
    #     y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
    #     y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
    #     y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))
    #
    # y = numpy.hstack((y, y_05))
    # y = numpy.vstack((y, y_09))
    # y = numpy.vstack((y, y_01))

    # plot_DCF(x, y, 'lambda', appendToTitle + 'SVM_minDCF_comparison')
