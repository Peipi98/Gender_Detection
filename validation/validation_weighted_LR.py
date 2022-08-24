# -*- coding: utf-8 -*-
import sys

import numpy
import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def validate_LR(scores, LR_labels, appendToTitle, l, pi=0.5):
    scores_append = np.hstack(scores)
    scores_tot_05 = compute_min_DCF(scores_append, LR_labels, 0.5, 1, 1)
    scores_tot_01 = compute_min_DCF(scores_append, LR_labels, 0.1, 1, 1)
    scores_tot_09 = compute_min_DCF(scores_append, LR_labels, 0.9, 1, 1)
    # plot_ROC(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l), 0.4)

    t = PrettyTable(["Type", "π=0.5", "π=0.1", "π=0.9"])
    t.title = appendToTitle
    t.add_row(['WEIGHTED_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    print(t)


def kfold_WEIGHTED_LR(DTR, LTR, l, appendToTitle, pi, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    PCA_LR_scores_append = []
    PCA2_LR_scores_append = []
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

        if zscore_Flag is True:
            D, Dte = znorm(D, Dte)

        if gauss_Flag is True:
            D_training = D
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D_training, Dte)


        scores = weighted_logistic_reg_score(D, L, Dte, l, pi)

        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

        if PCA_Flag is True:
            # PCA m=10
            P = PCA(D, L, m=10)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA_LR_scores = weighted_logistic_reg_score(DTR_PCA, L, DTE_PCA, l, pi)
            PCA_LR_scores_append.append(PCA_LR_scores)

            # PCA m=9
            P = PCA(D, L, m=9)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA2_LR_scores = weighted_logistic_reg_score(DTR_PCA, L, DTE_PCA, l)
            PCA2_LR_scores_append.append(PCA2_LR_scores)

    validate_LR(scores_append, LR_labels, appendToTitle, l, pi=pi)

    if PCA_Flag is True:
        validate_LR(PCA_LR_scores_append, LR_labels, appendToTitle + 'PCA_m10_', l, pi=pi)

        validate_LR(PCA2_LR_scores_append, LR_labels, appendToTitle + 'PCA_m9_', l, pi=pi)


def kfold_WEIGHTED_LR_calibration(DTR, LTR, l, PCA_Flag=False, gauss_Flag=False, zscore_Flag=False, pi=0.5):
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

        if zscore_Flag is True:
            D, Dte = znorm(D, Dte)

        if gauss_Flag is True:
            D_training = D
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D_training, Dte)

        scores = weighted_logistic_reg_score(D, L, Dte, l, pi)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), LR_labels


def validation_weighted_LR(DTR, LTR, L, appendToTitle, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
    for l in L:
        kfold_WEIGHTED_LR(DTR, LTR, l, appendToTitle, 0.5, PCA_Flag, gauss_Flag, zscore_Flag)
        kfold_WEIGHTED_LR(DTR, LTR, l, appendToTitle, 0.1, PCA_Flag, gauss_Flag, zscore_Flag)
        kfold_WEIGHTED_LR(DTR, LTR, l, appendToTitle, 0.9, PCA_Flag, gauss_Flag, zscore_Flag)

    x = numpy.logspace(-5, 1, 30)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    for xi in x:
        scores, labels = kfold_WEIGHTED_LR_calibration(DTR, LTR, xi, PCA_Flag, gauss_Flag, zscore_Flag)
        y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
        y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
        y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    plot_DCF(x, y, 'lambda', appendToTitle + 'WEIGHTED_LR_minDCF_comparison')
