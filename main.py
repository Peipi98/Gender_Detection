from evaluators.evaluation_LR import evaluation_LR
from evaluators.evaluation_MVG import evaluation_MVG
from plot_features import plot_features
from validators import *
import sys
sys.path.append("./evaluators")


if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    plot_features(DTR, LTR, 'RAW_')

    # RAW data

    print("############    MVG - RAW   ##############")
    evaluation_MVG(DTR, LTR, DTE, LTE, 'RAW_')

    print("############    Logistic Regression - RAW    ##############")
    L = [1e-6, 1e-4, 1e-2, 1.0]
    evaluation_LR(DTR, LTR, L, 'RAW_')
    
    # Gaussianization
    
    DTR = gaussianize_features(DTR, DTR)
    plot_features(DTR, LTR, 'GAUSSIANIZED_')

    print("############    MVG - gaussianization    ##############")
    evaluation_MVG(DTR, LTR, DTE, LTE, 'GAUSSIANIZED_')

    print("############    Logistic Regression - gaussianization    ##############")
    L = [1e-6, 1e-4, 1e-2]
    evaluation_LR(DTR, LTR, L, 'GAUSSIANIZED_')
