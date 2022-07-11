#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pylab

from validation_GMM import kfold_GMM
from validators import bayes_error_plot

def bayes_error_min_act_plot_GMM(D, LTE, pi, title, ylim):
    p = np.linspace(-3, 3, 21)
    pylab.title(title)
    pylab.plot(p, bayes_error_plot(p, D[0], LTE[0], minCost=False), color='r', label='actDCF')
    pylab.plot(p, bayes_error_plot(p, D[0], LTE[0], minCost=True), 'r--', label='minDCF')
    #pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False, th=-np.log(pi / (1-pi))), color='y')
    
    pylab.plot(p, bayes_error_plot(p, D[1], LTE[1], minCost=False), color='b', label='actDCF')
    pylab.plot(p, bayes_error_plot(p, D[1], LTE[1], minCost=True), 'b--', label='minDCF')
    #pylab.plot(p, bayes_error_plot(p, D, LTE, minCost=False, th=-np.log(pi / (1-pi))), color='y')
    
    pylab.ylim(0, ylim)
    pylab.savefig('../images/DCF_' + title + '.png')
    pylab.show()
    
def compute_bayes_plot():
    # Put here models to be compared
    
    bayes_error_min_act_plot_GMM([D1, D2], [L1,L2], 0.5, "", 0.4)