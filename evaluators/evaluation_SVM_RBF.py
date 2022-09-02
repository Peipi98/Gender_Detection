# -*- coding: utf-8 -*-
import sys

import numpy as np

sys.path.append('../')
from validators import *
from prettytable import PrettyTable


def evaluate_SVM_RBF(DTR, LTR, DTE, LTE, appendToTitle, C=1.0, K=1.0, gamma=0.01, PCA_Flag=False):
    SVM_labels = []
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    aStar, loss = train_SVM_RBF(DTR, LTR, C=C, K=K, gamma=gamma)

    kern = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K

    score = numpy.sum(numpy.dot(aStar * mrow(Z), kern), axis=0)

    SVM_labels = np.append(SVM_labels, LTE, axis=0)
    SVM_labels = np.hstack(SVM_labels)

    scores_append = np.hstack(score)
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.5, 1, 1)

    # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.5"
    t.add_row(['SVM_RBF, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.1, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.1"
    t.add_row(['SVM_RBF, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_min_DCF(scores_append, SVM_labels, 0.9, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = "minDCF: π=0.9"
    t.add_row(['SVM_RBF, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])

    print(t)

def svm_rbf_calibration(DTR, LTR, DTE, LTE, c, gamma):
    scores_append = []
    SVM_labels = []

    K = 1.0
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    aStar, loss = train_SVM_RBF(DTR, LTR, C=c, K=K, gamma=gamma)

    kern = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K

    scores = numpy.sum(numpy.dot(aStar * mrow(Z), kern), axis=0)
    scores_append.append(scores)

    SVM_labels = np.append(SVM_labels, LTE, axis=0)
    SVM_labels = np.hstack(SVM_labels)

    return np.hstack(scores_append), SVM_labels

def evaluation_SVM_RBF(DTR, LTR, DTE, LTE, K_arr, gamma_arr, appendToTitle, PCA_Flag=True):
    for K in K_arr:
        for gamma in gamma_arr:
            evaluate_SVM_RBF(DTR, LTR, DTE, LTE, appendToTitle, C=1.0, K=K, gamma=gamma, PCA_Flag=False)

    x = numpy.logspace(-4, 2, 15)   #x contains different values of C
    y = numpy.array([])
    gamma_minus_3 = numpy.array([])
    gamma_minus_2 = numpy.array([])
    gamma_minus_1 = numpy.array([])
    gamma_minus_0 = numpy.array([])

    for xi in x:
        print(xi)
        scores_gamma_minus_3, labels = svm_rbf_calibration(DTR, LTR, DTE, LTE, xi, 1e-3)
        scores_gamma_minus_2, _ = svm_rbf_calibration(DTR, LTR, DTE, LTE, xi, 1e-2)
        scores_gamma_minus_1, _ = svm_rbf_calibration(DTR, LTR, DTE, LTE, xi, 1e-1)
        scores_gamma_minus_0, _ = svm_rbf_calibration(DTR, LTR, DTE, LTE, xi, 1e-0)

        gamma_minus_3 = numpy.hstack((gamma_minus_3, bayes_error_plot_compare(0.5, scores_gamma_minus_3, labels)))
        gamma_minus_2 = numpy.hstack((gamma_minus_2, bayes_error_plot_compare(0.5, scores_gamma_minus_2, labels)))
        gamma_minus_1 = numpy.hstack((gamma_minus_1, bayes_error_plot_compare(0.5, scores_gamma_minus_1, labels)))
        gamma_minus_0 = numpy.hstack((gamma_minus_0, bayes_error_plot_compare(0.5, scores_gamma_minus_0, labels)))

    y = numpy.hstack((y, gamma_minus_3))
    y = numpy.vstack((y, gamma_minus_2))
    y = numpy.vstack((y, gamma_minus_1))
    y = numpy.vstack((y, gamma_minus_0))

    plot_DCF_for_SVM_RBF_calibration(x, y, 'C', appendToTitle + 'EVAL_SVM_RBF_minDCF_comparison')

