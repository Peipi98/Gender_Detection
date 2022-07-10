import sys

import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def kfold_SVM_polynomial(DTR, LTR, K, costant, appendToTitle, C=1.0, degree=2, PCA_Flag=True):
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
        print(i)

        aStar, loss = train_SVM_polynomial(D, L, C=C, constant=costant, degree=degree, K=K)
        Z = numpy.zeros(L.shape)
        Z[L == 1] = 1
        Z[L == 0] = -1
        kernel = (numpy.dot(D.T, Dte) + costant) ** degree + K * K
        scores = numpy.sum(numpy.dot(aStar * mrow(Z), kernel), axis=0)
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
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.5, 1, 1)

    #    plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM_POLY, K=' + str(K) + ', C=' + str(C), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.5"
    t.add_row(['SVM_POLY, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.1, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.1"
    t.add_row(['SVM_POLY, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.9"
    t.add_row(['SVM_POLY, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)


def single_F_POLY(D, L, C, K, costant=1.0, degree=2):
    nTrain = int(D.shape[1] * 0.8)
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    aStar, loss = train_SVM_polynomial(DTR, LTR, C=1.0, constant=costant, degree=degree, K=K)
    kernel = (numpy.dot(DTR.T, DTE) + costant) ** degree + K * K
    score = numpy.sum(numpy.dot(aStar * mrow(Z), kernel), axis=0)

    errorRate = (1 - numpy.sum((score > 0) == LTE) / len(LTE)) * 100
    print("K = %d, costant = %d, loss = %e, error =  %.1f" % (K, costant, loss, errorRate))

    scores_append = numpy.hstack(score)
    scores_tot = compute_min_DCF(scores_append, LTE, 0.5, 1, 1)
    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.5"
    t.add_row(['SVM, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)


def evaluation_SVM_polynomial(DTR, LTR, K_arr, C, appendToTitle, CON_array, PCA_Flag=True):
    # for costant in [1000]:
    #     for degree in [4]:
    #         for K in [1., 10.]:
    #             kfold_SVM_polynomial(DTR, LTR, K, costant, appendToTitle, C=1.0, degree=degree, PCA_Flag=False)
    #             single_F_POLY(DTR, LTR, C=1.0, K=K, costant=1000, degree=4)
    for costant in CON_array:
        for degree in [4]:
            for K in K_arr:
                kfold_SVM_polynomial(DTR, LTR, K, costant, appendToTitle, C=C, degree=degree, PCA_Flag=False)
                #single_F_POLY(DTR, LTR, C=C, K=K, costant=1000, degree=4)
