# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


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

        print('train')
        wStar, primal, dual, gap = train_SVM_linear(DTR, LTR, C=C, K=K)
        print('train_done')

        DTEEXT = numpy.vstack([Dte, K * numpy.ones((1, Dte.shape[1]))])
        scores = numpy.dot(wStar.T, DTEEXT).ravel()
        print(scores.shape)

        scores_append.append(scores)

        SVM_labels = np.append(SVM_labels, Lte, axis=0)
        SVM_labels = np.hstack(SVM_labels)

    scores_append = np.hstack(scores_append)
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.5, 1, 1)

#    plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)

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


def evaluation_SVM(DTR, LTR, K_arr, C_arr, appendToTitle):
    for K in K_arr:
        for C in C_arr:
            kfold_SVM(DTR, LTR, K, C, appendToTitle)
