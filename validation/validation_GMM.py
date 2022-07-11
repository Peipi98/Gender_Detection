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
    
    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["GMM_EM", round(llrs_tot, 3)])
    print(t)
    return round(llrs_tot, 3), round(llrs_tot_act, 3)
    
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
    plt.savefig('../images/GMM/' + title)
    plt.show()
    
    
def kfold_GMM(DTR, LTR, comp):
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

    llrs_tot_min, llrs_tot_act   =      validation_GMM("GMM full", 0.5, GMM_llrs, GMM_labels)
    llrsn_tot_min, llrsn_tot_act  =     validation_GMM("GMM diag", 0.5, GMM_llrsn, GMM_labels)
    llrst_tot_min, llrst_tot_act =      validation_GMM("GMM tied full", 0.5, GMM_llrst, GMM_labels)
    llrsnt_tot_min, llrsnt_tot_act  =   validation_GMM("GMM tied diag", 0.5, GMM_llrsnt, GMM_labels)
    
    llrs_min = [llrs_tot_min, llrsn_tot_min, llrst_tot_min, llrsnt_tot_min]
    llrs_act = [llrs_tot_act, llrsn_tot_act, llrst_tot_act, llrsnt_tot_act]
    
    return llrs_min, llrs_act, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels

def bayes_plot_bestGMM(title, pi, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels):
    bayes_error_min_act_plot(GMM_llrs, GMM_labels, 'GMM_full', 0.4)
    bayes_error_min_act_plot(GMM_llrsn, GMM_labels, 'GMM_diag', 0.4)
    bayes_error_min_act_plot(GMM_llrst, GMM_labels, 'GMM_tied', 0.4)
    bayes_error_min_act_plot(GMM_llrsnt, GMM_labels, 'GMM_tied_diag', 0.4)

        
if __name__ == "__main__":
    DTR, LTR = load("../Train.txt")
    DTR_gauss = gaussianize_features(DTR, DTR)
    DTR = stats.zscore(DTR, axis=1)
    score_raw_min = []
    score_gauss_min = []
    
    score_raw_act = []
    score_gauss_act = []
    
    # We'll train from 1 to 2^7 components
    '''
    # We'll train from 1 to 2^7 components
    componentsToTry=[0,1,2,3,4,5,6,7] 
    for comp in componentsToTry:
        print('RAW DATA')
        raw_min, raw_act, _ = kfold_GMM(DTR, LTR, comp)
        score_raw_min.append(raw_min)
        score_raw_act.append(raw_act)
        
        print('GAUSSIANIZED')
        gauss_min, gauss_act, _ = kfold_GMM(DTR_gauss, LTR, comp)
        score_gauss_min.append(gauss_min)
        score_gauss_act.append(gauss_act)
    
    print_minDCF_tables(score_raw_min, score_gauss_min, componentsToTry)
    print_act_DCF_tables(score_raw_act, score_gauss_act, componentsToTry)
    '''
    
    _, _, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels = kfold_GMM(DTR, LTR, 2)
    bayes_plot_bestGMM("prova", 0.5, GMM_llrs, GMM_llrsn, GMM_llrst, GMM_llrsnt, GMM_labels)
    
    