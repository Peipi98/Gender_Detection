# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import sys
sys.path.append("../")
from mlFunc import *

def compute_variance(X):
    mu = X.mean(0);
    Xc = X - mu
    X_2 = Xc ** 2
    X_sum = np.sum(X_2)
    
    variance = X_sum / (X.shape[0]-1)
    return variance

def compute_covariance(X,Y):
    
    mux = X.mean(0);
    muy = Y.mean(0);
    
    xc = X - mux
    yc = Y - muy
    
    cov = np.sum(X * Y.T) / (D.shape[1]-1)
    return cov

def compute_correlation(X,Y):
    x_sum = np.sum(X)
    y_sum = np.sum(Y)
    
    x2_sum = np.sum(X ** 2)
    y2_sum = np.sum(Y ** 2)
    
    sum_cross_prod = np.sum(X * Y.T)
    
    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum*y_sum
    denominator = np.sqrt((n * x2_sum - x_sum**2) * (n * y2_sum - y_sum**2))
    
    corr = numerator / denominator
    return corr
    
if __name__ == '__main__':
    DTR, LTR = load("../Train.txt")
    DTE, LTE = load("../Test.txt")
    
    corr = np.zeros((12,12))
    
    D = DTR[[0,1], :]
    X = DTR[0, :]
    Y = DTR[1, :]
    print()
    print(D.shape[1])
    print(compute_covariance(X,Y) )
    print(np.sum(X * Y.T))
    mu = D.mean(1)
    pearson_elem = np.abs(compute_covariance(X,Y) / (np.sqrt(compute_variance(X)) * np.sqrt(compute_variance(Y))))
    
    print(pearson_elem)
    
    print(compute_correlation(X, Y))
    
    for x in range(12):
        for y in range(12):
            D = DTR[[x,y], :]
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem
    
    #sns.heatmap(prova_corr)
    sns.set()
    heatmap = sns.heatmap(np.abs(corr),linewidth=0.2, cmap="Greys", square=True, cbar=False)
    
            
    #corr = np.hstack(corr)
    #corr = np.reshape(corr, (12,12))
    
    