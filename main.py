from mlFunc import *
from classifiers import *
from plot_features import plot_features
from validators import *
import sys
sys.path.append("./evaluators")
from mvg_script import *
from kfold_lr import *
from prettytable import PrettyTable

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    plot_features(DTR, LTR, 'RAW_')
    DTR = gaussianize_features(DTR, DTR)
    plot_features(DTR, LTR, 'GAUSSIANIZED_')
    
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    print("############    MVG - no gaussianization    ##############")

    kfold_MVG(DTR, LTR, DTE, LTE)
    print("############    Logistic Regression - no gaussianization    ##############")
    evaluation_LR()





