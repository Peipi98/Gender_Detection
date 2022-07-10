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
from Classifiers.GMM import GMM
import scipy.stats as stats

def validation_GMM(title, pi, GMM_llrs, LTE):
    GMM_llrs = np.hstack(GMM_llrs)
    llrs_tot = compute_min_DCF(GMM_llrs, LTE, pi, 1, 1)

    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["GMM_EM", round(llrs_tot, 3)])
    print(t)
    return round(llrs_tot, 3)
    
def ll_GMM(D, L, Dte, Lte, llr, cov, comp, i):
    #CLASS PRIORS: WE CONSIDER A BALANCED APPLICATION
    prior_0 = 0.5
    prior_1 = 0.5
    
    #GMM MODELS
    
    #optimal_m = 10
    optimal_comp = comp
    optimal_cov = cov
    optimal_alpha = 0.1
    optimal_psi = 0.01
    
    gmm = GMM(D, L, Dte, Lte, [prior_0, prior_1], iterations=optimal_comp, alpha=optimal_alpha, psi=optimal_psi, typeOfGmm=optimal_cov)

    gmm.train()
    gmm.test()
    
    #llr.append(gmm.llrs)
    llr = np.append(llr, gmm.llrs)
    llr = np.hstack(llr)
    return llr

def print_minDCF_tables(score_raw, score_gauss, components):
    types = ['full-cov', 'diag-cov', 'tied full-cov', 'tied diag-cov']
    
    header = ['']
    print(np.shape(score_raw))
    print(score_raw)
    score_raw = np.reshape(np.hstack(score_raw), ((components+1), 4)).T
    score_gauss = np.reshape(np.hstack(score_gauss), ((components+1), 4)).T

    print(np.shape(score_raw))
    for i in range(components+1):
        header.append(2 ** i)
        
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

def plot_minDCF_GMM(score_raw, score_gauss, title, components):
    labels = []
    
    for i in range(components+1):
        labels.append(2 ** i)
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, score_raw, width, label='Raw')
    rects2 = ax.bar(x + width/2, score_gauss, width, label='Gaussianized')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('DCF')
    ax.set_title(title)
    ax.set_xticks(x, labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    plt.savefig('../images/GMM/' + title)
    #plt.show()
    
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

    llrs_tot    =     validation_GMM("GMM full", 0.5, GMM_llrs, GMM_labels)
    llrsn_tot   =     validation_GMM("GMM diag", 0.5, GMM_llrsn, GMM_labels)
    llrst_tot   =     validation_GMM("GMM tied full", 0.5, GMM_llrst, GMM_labels)
    llrsnt_tot   =    validation_GMM("GMM tied diag", 0.5, GMM_llrsnt, GMM_labels)
    
    return [llrs_tot, llrsn_tot, llrst_tot, llrsnt_tot]
        
if __name__ == "__main__":
    DTR, LTR = load("../Train.txt")
    DTR_gauss = gaussianize_features(DTR, DTR)
    DTR = stats.zscore(DTR, axis=1)
    score_raw = []
    score_gauss = []
    
    components = 7
    # We'll train from 1 to 2^7 components
    for comp in range(components+1):
        print('RAW DATA')
        score_raw.append(kfold_GMM(DTR, LTR, comp))
        print('GAUSSIANIZED')
        score_gauss.append(kfold_GMM(DTR_gauss, LTR, comp))
    
    print_minDCF_tables(score_raw, score_gauss, components)