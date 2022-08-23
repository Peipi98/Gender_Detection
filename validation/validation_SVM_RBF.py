import sys

import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def kfold_SVM_RBF(DTR, LTR, appendToTitle, C=1.0, K=1, gamma=1, PCA_Flag=False, gauss_Flag=False, zscore_Flag=False):
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
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.5, 1, 1)
    #act_DCF_05 = compute_act_DCF(scores_append, SVM_labels, 0.5, 1, 1, )

    plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM_RFB, K=' + str(K) + ', C=' + str(C), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.5"
    t.add_row(['SVM_RFB, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.1, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.1"
    t.add_row(['SVM_RFB, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.9"
    t.add_row(['SVM_RFB, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)


def single_F_RFB(D, L, C, K, gamma):
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

    aStar, loss = train_SVM_RBF(DTR, LTR, C=1.0, K=K, gamma=gamma)

    kern = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K

    score = numpy.sum(numpy.dot(aStar * mrow(Z), kern), axis=0)

    errorRate = (1 - numpy.sum((score > 0) == LTE) / len(LTE)) * 100
    print("K = %d, gamma = %.1f, loss = %e, error =  %.1f" % (K, gamma, loss, errorRate))
    scores_append = numpy.hstack(score)
    scores_tot = compute_min_DCF(scores_append, LTE, 0.5, 1, 1)
    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.5"
    t.add_row(['SVM, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)

def kfold_svm_rbf_calibration(DTR, LTR, c, gamma):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    SVM_labels = []
    K = 1.0

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

        Z = L * 2 - 1
        aStar, loss = train_SVM_RBF(D, L, C=c, K=K, gamma=gamma)
        kern = numpy.zeros((D.shape[1], Dte.shape[1]))
        for i in range(D.shape[1]):
            for j in range(Dte.shape[1]):
                kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
        scores = numpy.sum(numpy.dot(aStar * mrow(Z), kern), axis=0)
        scores_append.append(scores)

        SVM_labels = np.append(SVM_labels, Lte, axis=0)
        SVM_labels = np.hstack(SVM_labels)

    return np.hstack(scores_append), SVM_labels



def validation_SVM_RFB(DTR, LTR, K_arr, gamma_arr, appendToTitle, PCA_Flag=True, gauss_Flag=False, zscore_Flag=False):
    # for K in K_arr:
    #     for gamma in gamma_arr:
    #         kfold_SVM_RBF(DTR, LTR, appendToTitle, C=1.0, K=K, gamma=gamma, PCA_Flag=False, gauss_Flag=gauss_Flag, zscore_Flag=zscore_Flag)
    #         #single_F_RFB(DTR, LTR, C=1.0, K=1.0, gamma=gamma)
    x = numpy.logspace(-4, 2, 12)  # x contains different values of C
    count = 1
    y = numpy.array([])
    gamma_minus_3 = numpy.array([])
    gamma_minus_2 = numpy.array([])
    gamma_minus_1 = numpy.array([])
    gamma_minus_0 = numpy.array([])

    for xi in x:
        print("-------------------------------------")
        print("punto " + str(count) + " di " + str(x.shape[0]))
        count += 1
        print("1e-3")
        scores_gamma_minus_3, labels = kfold_svm_rbf_calibration(DTR, LTR, xi, 1e-3)
        print("1e-2")
        scores_gamma_minus_2, _ = kfold_svm_rbf_calibration(DTR, LTR, xi, 1e-2)
        print("1e-1")
        scores_gamma_minus_1, _ = kfold_svm_rbf_calibration(DTR, LTR, xi, 1e-1)
        print("1e-0")
        scores_gamma_minus_0, _ = kfold_svm_rbf_calibration(DTR, LTR, xi, 1e-0)

        gamma_minus_3 = numpy.hstack((gamma_minus_3, bayes_error_plot_compare(0.5, scores_gamma_minus_3, labels)))
        gamma_minus_2 = numpy.hstack((gamma_minus_2, bayes_error_plot_compare(0.5, scores_gamma_minus_2, labels)))
        gamma_minus_1 = numpy.hstack((gamma_minus_1, bayes_error_plot_compare(0.5, scores_gamma_minus_1, labels)))
        gamma_minus_0 = numpy.hstack((gamma_minus_0, bayes_error_plot_compare(0.5, scores_gamma_minus_0, labels)))

    y = numpy.hstack((y, gamma_minus_3))
    y = numpy.vstack((y, gamma_minus_2))
    y = numpy.vstack((y, gamma_minus_1))
    y = numpy.vstack((y, gamma_minus_0))

    plot_DCF_for_SVM_RBF_calibration(x, y, 'C', appendToTitle + 'VAL_SVM_RBF_minDCF_comparison')