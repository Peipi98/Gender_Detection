# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append("../")
from mlFunc import *

if __name__ == '__main__':
    DTR, LTR = load("../Train.txt")
    DTE, LTE = load("../Test.txt")
    
    print(DTR[:][1])
    
    for i in range(12):
        #plot_histogram(DTR[:][i], LTR, ["male", "female"], )
        labels = ["male", "female"]
        title = "feature_" + str(i)
        matplotlib.pyplot.figure()
        matplotlib.pyplot.title(title)
        
        y = DTR[:, LTR == 0][i]
        matplotlib.pyplot.hist(y, bins=60, density=True, alpha=0.4, linewidth = 1.0, color = 'red', edgecolor = 'black', label=labels[0])
        y = DTR[:, LTR == 1][i]
        matplotlib.pyplot.hist(y, bins=60, density=True, alpha=0.4, linewidth = 1.0, color = 'blue', edgecolor = 'black', label=labels[1])
        matplotlib.pyplot.legend()
        plt.savefig('../images/hist_' + title + '.png')
        matplotlib.pyplot.show()
        