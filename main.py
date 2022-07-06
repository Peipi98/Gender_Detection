from evaluators.kfold_lr import evaluation_LR
from evaluators.mvg_script import kfold_MVG
from plot_features import plot_features
from validators import *
import sys
sys.path.append("./evaluators")


if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    plot_features(DTR, LTR, 'RAW_')
    
    # RAW data
    
    print("############    MVG - no gaussianization    ##############")
    kfold_MVG(DTR, LTR, DTE, LTE)
    
    print("############    Logistic Regression - no gaussianization    ##############")
    evaluation_LR(DTR, LTR)
    
    # Gaussianization
    
    DTR = gaussianize_features(DTR, DTR)
    plot_features(DTR, LTR, 'GAUSSIANIZED_')

    print("############    MVG - gaussianization    ##############")
    kfold_MVG(DTR, LTR, DTE, LTE)
    
    print("############    Logistic Regression - gaussianization    ##############")
    evaluation_LR(DTR, LTR)



