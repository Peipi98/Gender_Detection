# -*- coding: utf-8 -*-
import sys

import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def calibrate_scores(scores, labels):

    scores_70 = scores[:int(len(scores) * 0.7)]
    scores_30 = scores[int(len(scores) * 0.7):]
    labels_70 = labels[:int(len(labels) * 0.7)]
    labels_30 = labels[int(len(labels) * 0.7):]

    #logreg = WeighLogReg(numpy.array([scores_70]), labels_70, numpy.array([scores_30]), labels_30, 10 ** -3)
    S, estimated_w, estimated_b = logistic_reg_calibration(numpy.array([scores_70]), labels_70, numpy.array([scores_30]), 10**-3)

    return numpy.array(S), labels_30, estimated_w, estimated_b


def kfold_SVM(DTR, LTR, K, C, appendToTitle):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
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
        print(i)
        wStar, primal, dual, gap = train_SVM_linear(D, L, C=C, K=K)

        DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])
        scores = numpy.dot(wStar.T, DTEEXT).ravel()
        scores_append.append(scores)

        SVM_labels = np.append(SVM_labels, Lte, axis=0)
        SVM_labels = np.hstack(SVM_labels)


    scores_append = np.hstack(scores_append)
    cal_scores, cal_labels, w, b = calibrate_scores(scores_append, SVM_labels)


    #    plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    bayes_error_min_act_plot(cal_scores, cal_labels, appendToTitle + 'SVM_CALIBRATED_, K=' + str(K) + ', C=' + str(C), 0.4)

    # def compute_confusion_matrix_binary(labels, llrs, pi, Cfn, Cfp, t=None):
    #
    #     if (t == None):
    #         t = -numpy.log((pi * Cfn) / ((1 - pi) * Cfp))
    #
    #     # creo la confusion matrix
    #     confusion_matrix = numpy.zeros((2, 2))
    #
    #     indexes_label_0 = (labels == 0)
    #     indexes_label_1 = (labels == 1)
    #
    #     confusion_matrix[0][0] = (llrs[indexes_label_0] <= t).sum()
    #     confusion_matrix[0][1] = (llrs[indexes_label_1] <= t).sum()
    #
    #     confusion_matrix[1][1] = (llrs[indexes_label_1] > t).sum()
    #     confusion_matrix[1][0] = (llrs[indexes_label_0] > t).sum()
    #
    #     return confusion_matrix
    #
    # confusion_matrix = compute_confusion_matrix_binary(numpy.array(cal_labels), numpy.array(cal_scores), 0.5, 1, 1)
    # Bt = 0.5 * (confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1]))
    #
    # Bf = 0.5 * (confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0]))
    # calibratedDcf = Bt + Bf / 0.5
    #print(calibratedDcf)


def calibrate_SVM(DTR, LTR, appendToTitle=''):
    K = 10.0
    C = 1.0
    kfold_SVM(DTR, LTR, K, C, appendToTitle)
