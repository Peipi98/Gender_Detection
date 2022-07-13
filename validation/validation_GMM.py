# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
sys.path.append('../')
from mlFunc import *
from validators import *
from classifiers import *
from prettytable import PrettyTable
from Classifiers.GMM import GMM_Full
import scipy.stats as stats

def validation_GMM(title, pi, GMM_llrs, LTE):
    GMM_llrs = np.hstack(GMM_llrs)
    llrs_tot = compute_min_DCF(GMM_llrs, LTE, pi, 1, 1)
    llrs_tot_act = compute_act_DCF(GMM_llrs, LTE, pi, 1, 1)
    llrs_tot_xvd = compute_act_DCF(GMM_llrs, LTE, pi, 1, 1, -np.log(pi / (1-pi)))
    
    # t = PrettyTable(["Type", "minDCF"])
    # t.title = title
    # t.add_row(["GMM_EM", round(llrs_tot, 3)])
    # print(t)
    return round(llrs_tot, 3), round(llrs_tot_act, 3), round(llrs_tot_xvd, 3)
    
def ll_GMM(D, L, Dte, Lte, llr, cov, comp, i):
    # GMM_llrs, 'full', comp, i
    #CLASS PRIORS: WE CONSIDER A BALANCED APPLICATION
    
    #GMM MODELS
    # π = 0.5
    
    #optimal_m = 10
    optimal_comp = comp
    optimal_cov = cov
    optimal_alpha = 0.1
    optimal_psi = 0.01
    
    llr.extend(GMM_Full(D, Dte, L, optimal_alpha, 2 ** optimal_comp, optimal_cov).tolist())
    return llr

def print_minDCF_tables(score_raw, score_gauss, components):
    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']
    
    header = ['']
    n_comp = len(components)
    print(np.shape(score_raw))
    print(score_raw)
    score_raw = np.reshape(np.hstack(score_raw), (n_comp, 4)).T
    score_gauss = np.reshape(np.hstack(score_gauss), (n_comp, 4)).T
    
    comp = np.exp2(components).astype(int).tolist()

    print(np.shape(score_raw))
    for i in comp:
        header.append(i)
    print(header)
    for i in range(len(types)):
        t1 = PrettyTable(header)
        
        t1.title = types[i]
        
        raw_full = score_raw[i].tolist()
        gauss_full = score_gauss[i].tolist()
        
        raw_full.insert(0,'raw')
        gauss_full.insert(0,'gaussianized')
        t1.add_row(raw_full)
        t1.add_row(gauss_full)
        print(t1)
        plot_minDCF_GMM(score_raw[i].tolist(), score_gauss[i].tolist(), types[i], components)
        
def print_act_DCF_tables(score_raw, score_gauss, components):
    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']
    
    header = ['']
    n_comp = len(components)
    print(np.shape(score_raw))
    print(score_raw)
    score_raw = np.reshape(np.hstack(score_raw), ((n_comp), 4)).T
    score_gauss = np.reshape(np.hstack(score_gauss), ((n_comp), 4)).T
    
    comp = np.exp2(components).astype(int).tolist()

    
    print(np.shape(score_raw))
    for i in comp:
        header.append(i)
    print(header)
    for i in range(len(types)):
        t1 = PrettyTable(header)
        
        t1.title = types[i]
        
        raw_full = score_raw[i].tolist()
        gauss_full = score_gauss[i].tolist()
        
        raw_full.insert(0,'raw')
        gauss_full.insert(0,'gaussianized')
        t1.add_row(raw_full)
        t1.add_row(gauss_full)
        print(t1)

def plot_minDCF_GMM(score_raw, score_gauss, title, components):
    labels = np.exp2(components).astype(int)
    
    # for i in range(components):
    #     labels.append(2 ** (i+1))
    
    
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(x - 0.2, score_raw, width, label = 'Raw')
    plt.bar(x + 0.2, score_gauss, width, label = 'Gaussianized')
      
    plt.xticks(x, labels)
    plt.ylabel("DCF")
    plt.title(title)
    plt.legend()
    plt.savefig('./images/GMM/' + title)
    plt.show()
    
    
def kfold_GMM(DTR, LTR, pi, comp, zscore = False, Gauss_flag = False):
    k = 5
    Dtr = np.split(DTR, k, axis=1)
    Ltr = np.split(LTR, k)

    GMM_llrs = []
    GMM_llrsn = []
    GMM_llrst = []
    GMM_llrsnt = []
    GMM_labels = []

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

        if (zscore):
            D, Dte = znorm(D, Dte)

        if(Gauss_flag):
            D_training = D
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D_training, Dte)

        print("components: " + str(comp) + " | fold " + str(i))
        GMM_labels = np.append(GMM_labels, Lte)
        GMM_labels = np.hstack(GMM_labels)
        
        # RAW DATA
        
        # full-cov
        GMM_llrs = ll_GMM(D, L, Dte, Lte, GMM_llrs, 'full', comp, i)
        
        # diag-cov
        GMM_llrsn = ll_GMM(D, L, Dte, Lte, GMM_llrsn, 'diag', comp, i)
        
        # full-cov tied
        GMM_llrst = ll_GMM(D, L, Dte, Lte, GMM_llrst, 'tied_full', comp, i)
        
        # diag-cov tied
        GMM_llrsnt = ll_GMM(D, L, Dte, Lte, GMM_llrsnt, 'tied_diag', comp, i)

    llrs_tot_min, llrs_tot_act, llrs_tot_xvd   =      validation_GMM("GMM full", pi, GMM_llrs, GMM_labels)
    llrsn_tot_min, llrsn_tot_act, llrsn_tot_xvd  =     validation_GMM("GMM diag", pi, GMM_llrsn, GMM_labels)
    llrst_tot_min, llrst_tot_act, llrst_tot_xvd =      validation_GMM("GMM tied full", pi, GMM_llrst, GMM_labels)
    llrsnt_tot_min, llrsnt_tot_act, llrsnt_tot_xvd  =   validation_GMM("GMM tied diag", pi, GMM_llrsnt, GMM_labels)
    
    llrs_min = [llrs_tot_min, llrsn_tot_min, llrst_tot_min, llrsnt_tot_min]
    llrs_act = [llrs_tot_act, llrsn_tot_act, llrst_tot_act, llrsnt_tot_act]
    llrs_xvd = [llrs_tot_xvd, llrsn_tot_xvd, llrst_tot_xvd, llrsnt_tot_xvd]
    return llrs_min, llrs_act, llrs_xvd, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels

def bayes_error_min_act_plot_GMM(D, LTE, pi, title, ylim):
    p = numpy.linspace(-3, 3, 21)
    pylab.title(title)
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False), color='r', label='actDCF')
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=True), 'r--', label='minDCF')
    pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False, th=-np.log(pi / (1-pi))), color='y', label='theoretical')
    pylab.ylim(0, ylim)
    pylab.legend()
    pylab.savefig('./images/DCF_' + title + '.png')
    pylab.show()

def bayes_plot_bestGMM(title, width, pi, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels):
    bayes_error_min_act_plot_GMM(GMM_llrs, GMM_labels, pi, title + 'GMM_full', width)
    bayes_error_min_act_plot_GMM(GMM_llrsn, GMM_labels, pi, title + 'GMM_diag', width)
    bayes_error_min_act_plot_GMM(GMM_llrst, GMM_labels, pi, title + 'GMM_tied', width)
    bayes_error_min_act_plot_GMM(GMM_llrsnt, GMM_labels, pi, title + 'GMM_tied_diag', width)
'''
def ROC_GMM():
 '''   
        
def validation_GMM_tot(DTR, LTR, pi, zscore=False, gauss=False):
    score_raw_min = []
    score_gauss_min = []
    
    score_raw_act = []
    score_gauss_act = []
    
    score_raw_xvd = []
    score_gauss_xvd = []
    # We'll train from 1 to 2^7 components
    
    # We'll train from 1 to 2^7 components
    componentsToTry=[1,2,3,4,5,6,7]
    for comp in componentsToTry:

        print('RAW DATA')
        raw_min, raw_act, raw_xvd, *_ = kfold_GMM(DTR, LTR, pi, comp, zscore=zscore, Gauss_flag=False)
        score_raw_min.append(raw_min)
        score_raw_act.append(raw_act)
        score_raw_xvd.append(raw_xvd)

        print('GAUSSIANIZED')
        gauss_min, gauss_act, gauss_xvd, *_ = kfold_GMM(DTR, LTR, pi, comp, zscore=zscore, Gauss_flag=True)
        score_gauss_min.append(gauss_min)
        score_gauss_act.append(gauss_act)
        score_gauss_xvd.append(gauss_xvd)
    print("======= min DCF =======")
    print_minDCF_tables(score_raw_min, score_gauss_min, componentsToTry)
    print("======= act DCF =======")
    print_act_DCF_tables(score_raw_act, score_gauss_act, componentsToTry)
    print("======= theoretical =======")
    print_act_DCF_tables(score_raw_xvd, score_gauss_xvd, componentsToTry)

    
def validation_GMM_ncomp(DTR, LTR, pi, n, zscore=False, gauss=False):
    raw_min, raw_act, raw_x, GMM_llrs_raw, GMM_llrsn_raw, GMM_llrst_raw, GMM_llrsnt_raw, GMM_labels_raw = kfold_GMM(DTR, LTR, pi, n, zscore=zscore, Gauss_flag=gauss)
    gauss_min, gauss_act, gauss_x, GMM_llrs_g, GMM_llrsn_g, GMM_llrst_g, GMM_llrsnt_g, GMM_labels_g = kfold_GMM(DTR, LTR, pi, n, zscore=zscore, Gauss_flag=True)

    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']
    t = PrettyTable(["", 'minDCF', 'actDCF', 'theoretical'])
    t.title = "GMM π=" + str(pi)
    for i in range(len(raw_min)):
        t.add_row(["raw " + types[i], raw_min[i], raw_act[i], raw_x[i]])
    for i in range(len(gauss_min)):
        t.add_row(['gaussian. ' + types[i], gauss_min[i], gauss_act[i], gauss_x[i]])
    print(t)
    bayes_plot_bestGMM("raw_", 0.4, pi, GMM_llrs_raw, GMM_llrsn_raw, GMM_llrst_raw, GMM_llrsnt_raw, GMM_labels_raw)
    plot_ROC(GMM_llrs_raw, GMM_labels_raw, 'GMM_full_raw')
    plot_ROC(GMM_llrsn_raw, GMM_labels_raw, 'GMM_diag_raw')
    plot_ROC(GMM_llrst_raw, GMM_labels_raw, 'GMM_tied_raw')
    plot_ROC(GMM_llrsnt_raw, GMM_labels_raw, 'GMM_tied_diag_raw')

    bayes_plot_bestGMM("gauss_", 1, pi, GMM_llrs_g, GMM_llrsn_g, GMM_llrst_g, GMM_llrsnt_g, GMM_labels_g)
    plot_ROC(GMM_llrs_g, GMM_labels_g, 'GMM_full_raw')
    plot_ROC(GMM_llrsn_g, GMM_labels_g, 'GMM_diag_raw')
    plot_ROC(GMM_llrst_g, GMM_labels_g, 'GMM_tied_raw')
    plot_ROC(GMM_llrsnt_g, GMM_labels_g, 'GMM_tied_diag_raw')

