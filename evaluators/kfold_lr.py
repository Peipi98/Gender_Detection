# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def kfold_LR(l):
    DTR, LTR = load("../Train.txt")
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

        scores = linear_reg_score(D, L, Dte, l)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    scores_append = np.hstack(scores_append)
    scores_tot = compute_min_DCF(scores_append, LR_labels, 0.5, 1, 1)

    plot_ROC(scores_append, LR_labels, 'LR, lambda=' + str(l))

    # Cfn and Ctp are set to 1
    bayes_error_min_act_plot(scores_append, LR_labels, 'LR, lambda=' + str(l), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.5"
    t.add_row(['LR, lambda=' + str(l), scores_tot])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_min_DCF(scores_append, LR_labels, 0.1, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.1"
    t.add_row(['LR, lambda=' + str(l), scores_tot])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_min_DCF(scores_append, LR_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.9"
    t.add_row(['LR, lambda=' + str(l), scores_tot])

    print(t)


if __name__ == '__main__':
    for l in [1e-6, 1e-4, 1e-2, 1.0]:
        kfold_LR(l)
