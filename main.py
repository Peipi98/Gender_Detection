from mlFunc import *
from classifiers import *
from plot_features import plot_features
from validators import *
from prettytable import PrettyTable

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    #DTE, LTE = load("Test.txt")
    plot_features(DTR, LTR, 'RAW_')
    DTR = gaussianize_features(DTR, DTR)
    plot_features(DTR, LTR, 'GAUSSIANIZED_')






