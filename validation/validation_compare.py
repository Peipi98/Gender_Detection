# -*- coding: utf-8 -*-
import sys

import numpy
import numpy as np

from evaluators.compare_best_2 import bayes_error_plot_2best, ROC_2best
from evaluators.evaluation_GMM import evaluation_GMM_ncomp, evaluation_scores_GMM_ncomp

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

def compare(scores, scores2, LTE):
    scores = np.hstack(scores)
    scores2 = np.hstack(scores2)


    bayes_error_plot_2best([scores, scores2], LTE, 0.5, '', 0.4)
    ROC_2best([scores, scores2], LTE, 0.5, '')



def kfold_validation_compare(DTR, LTR, l=None):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    scores2_append = []
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

        if l is not None:
            # GMM_llrst_raw = evaluation_scores_GMM_ncomp('', D, L, Dte, Lte, 0.5, 2)
            # scores_append.append(GMM_llrst_raw)
            Z = L * 2 - 1
            C = 1.0
            K = 1.0
            aStar, loss = train_SVM_RBF(D, L, C=C, K=K, gamma=0.001)
            kern = numpy.zeros((D.shape[1], Dte.shape[1]))
            for i in range(D.shape[1]):
                for j in range(Dte.shape[1]):
                    kern[i, j] = numpy.exp(-0.001 * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
            scores = numpy.sum(numpy.dot(aStar * mrow(Z), kern), axis=0)

            scores_append.append(scores)



        #GMM_llrst_raw = evaluation_scores_GMM_ncomp('', D, L, Dte, Lte, 0.5, 2)
        _, _, llrs_tied = tied_cov_GC(Dte, D, L)
        scores2_append.append(llrs_tied)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    scores_append = np.hstack(scores_append)
    cal_scores, cal_labels, w, b = calibrate_scores(scores_append, LR_labels)
    scores_append = scores_append.reshape((1, 6000))
    final_score = numpy.dot(w.T, scores_append) + b
    compare(final_score, scores2_append, LR_labels)


def compare_2_validation(DTR, LTR, L):
    for l in L:
        kfold_validation_compare(DTR, LTR, l)
