# -*- coding: utf-8 -*-
import sys

import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def validate_LR(scores, LR_labels, appendToTitle, l):
    scores_append = np.hstack(scores)
    scores_tot = compute_min_DCF(scores_append, LR_labels, 0.5, 1, 1)

    # plot_ROC(scores_append, LR_labels, appendToTitle + 'QUAD_LR, lambda=' + str(l))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, LR_labels, appendToTitle + 'QUAD_LR, lambda=' + str(l), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "minDCF: π=0.5"
    t.add_row(['QUAD_LR, lambda=' + str(l), round(scores_tot, 3)])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_min_DCF(scores_append, LR_labels, 0.1, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "minDCF: π=0.1"
    t.add_row(['QUAD_LR, lambda=' + str(l), round(scores_tot, 3)])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_min_DCF(scores_append, LR_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "minDCF: π=0.9"
    t.add_row(['QUAD_LR, lambda=' + str(l), round(scores_tot, 3)])

    print(t)


def kfold_QUAD_LR(DTR, LTR, l, appendToTitle, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
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

        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]

        if zscore_Flag is True:
            D, Dte = znorm(D, Dte)

        if gauss_Flag is True:
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D, Dte)

        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, D)
        expanded_DTE = numpy.apply_along_axis(vecxxT, 0, Dte)
        phi = numpy.vstack([expanded_DTR, D])

        phi_DTE = numpy.vstack([expanded_DTE, Dte])

        scores = quad_logistic_reg_score(phi, L, phi_DTE, l)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    validate_LR(scores_append, LR_labels, appendToTitle, l)


def kfold_QUAD_LR_calibration(DTR, LTR, l, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
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

        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]

        if zscore_Flag is True:
            D, Dte = znorm(D, Dte)

        if gauss_Flag is True:
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D, Dte)

        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, D)
        expanded_DTE = numpy.apply_along_axis(vecxxT, 0, Dte)
        phi = numpy.vstack([expanded_DTR, D])

        phi_DTE = numpy.vstack([expanded_DTE, Dte])

        scores = quad_logistic_reg_score(phi, L, phi_DTE, l)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), LR_labels


def validation_quad_LR(DTR, LTR, L, appendToTitle, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
    for l in L:
        kfold_QUAD_LR(DTR, LTR, l, appendToTitle, PCA_Flag, gauss_Flag, zscore_Flag)

    x = numpy.logspace(-5, 1, 30)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    for xi in x:
        scores, labels = kfold_QUAD_LR_calibration(DTR, LTR, xi, PCA_Flag, gauss_Flag, zscore_Flag)
        y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
        y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
        y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    #plot_DCF(x, y, 'lambda', appendToTitle + 'QUAD_LR_minDCF_comparison')
