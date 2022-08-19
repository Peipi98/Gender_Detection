# -*- coding: utf-8 -*-
import sys

import numpy as np

sys.path.append('../')
from validators import *


def calibrate_scores(scores, labels):
    scores_70 = scores[:int(len(scores) * 0.7)]
    scores_30 = scores[int(len(scores) * 0.7):]
    labels_70 = labels[:int(len(labels) * 0.7)]
    labels_30 = labels[int(len(labels) * 0.7):]

    S, estimated_w, estimated_b = logistic_reg_calibration(numpy.array([scores_70]), labels_70,
                                                           numpy.array([scores_30]), 1e-3)

    return numpy.array(S), labels_30, estimated_w, estimated_b

def kfold_SVM_RFB(DTR, LTR, appendToTitle, C=1.0, K=1, gamma=1, PCA_Flag=False, gauss_Flag=False, zscore_Flag=False):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    PCA_SVM_scores_append = []
    PCA2_SVM_scores_append = []
    SVM_labels = []

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
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D, Dte)

        print(i)

        Z = L * 2 - 1
        aStar, loss = train_SVM_RBF(D, L, C=C, K=K, gamma=gamma)
        kern = numpy.zeros((D.shape[1], Dte.shape[1]))
        for i in range(D.shape[1]):
            for j in range(Dte.shape[1]):
                kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
        scores = numpy.sum(numpy.dot(aStar * mrow(Z), kern), axis=0)

        scores_append.append(scores)

        SVM_labels = np.append(SVM_labels, Lte, axis=0)
        SVM_labels = np.hstack(SVM_labels)

        if PCA_Flag is True:
            # PCA m=10
            P = PCA(D, L, m=10)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA_SVM_scores = 0  # todo
            PCA_SVM_scores_append.append(PCA_SVM_scores)

            # PCA m=9
            P = PCA(D, L, m=9)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA2_SVM_scores = 0  # todo
            PCA2_SVM_scores_append.append(PCA2_SVM_scores)

    scores_append = np.hstack(scores_append)
    cal_scores, cal_labels, w, b = calibrate_scores(scores_append, SVM_labels)
    scores_append = scores_append.reshape((1, 6000))
    final_score = numpy.dot(w.T, scores_append) + b

    # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    bayes_error_min_act_plot(final_score, SVM_labels, appendToTitle + 'RAW_, K=' + str(K) + ', C=' + str(C), 0.4)


def SVM_RBF_score_calibration(DTR, LTR, K_arr, gamma_arr, appendToTitle, PCA_Flag=False, gauss_Flag=False, zscore_Flag=False):
    for K in K_arr:
        for gamma in gamma_arr:
            kfold_SVM_RFB(DTR, LTR, appendToTitle, C=1.0, K=K, gamma=gamma, PCA_Flag=False, gauss_Flag=gauss_Flag,
                          zscore_Flag=zscore_Flag)
