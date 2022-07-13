# -*- coding: utf-8 -*-
import sys

import numpy
import numpy as np

from evaluators.compare_best_2 import bayes_error_plot_2best, ROC_2best
from evaluators.evaluation_GMM import evaluation_GMM_ncomp, evaluation_scores_GMM_ncomp

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


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
            GMM_llrst_raw = evaluation_scores_GMM_ncomp('', D, L, Dte, Lte, 0.5, 2)
            scores_append.append(GMM_llrst_raw)

        #GMM_llrst_raw = evaluation_scores_GMM_ncomp('', D, L, Dte, Lte, 0.5, 2)
        _, _, llrs_tied = tied_cov_GC(Dte, D, L)
        scores2_append.append(llrs_tied)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    compare(scores_append, scores2_append, LR_labels)


def compare_2_validation(DTR, LTR, L):
    for l in L:
        kfold_validation_compare(DTR, LTR, l)
