import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
import sys
sys.path.append("./functions")
from mlFunc import *

if __name__ == "__main__":
    D, L = load("Train.txt")
    print(D.shape)
    print(L.shape)