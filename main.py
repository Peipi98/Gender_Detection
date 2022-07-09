from validation.evaluation_LR import evaluation_LR
from validation.evaluation_MVG import evaluation_MVG
from validation.evaluation_SVM import evaluation_SVM
from plot_features import plot_features
from validators import *
import sys
import scipy.stats as stats

sys.path.append("./evaluators")

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    #    plot_features(DTR, LTR, 'RAW_')

    # RAW data

    print("############    MVG - RAW   ##############")
#    evaluation_MVG(DTR, LTR, DTE, LTE, 'RAW_')

    print("############    Logistic Regression - RAW    ##############")
    L = [1e-6, 1e-4, 1e-2, 1.0]

#    evaluation_LR(DTR, LTR, L, 'RAW_')

    print("############    Support Vector Machine - RAW    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
#    evaluation_SVM(DTR, LTR, K_arr, C_arr, 'RAW_')

    # Gaussianization

##    DTR = gaussianize_features(DTR, DTR)
    #    plot_features(DTR, LTR, 'GAUSSIANIZED_')

    print("############    MVG - gaussianization    ##############")
#    evaluation_MVG(DTR, LTR, DTE, LTE, 'GAUSSIANIZED_')

    print("############    Logistic Regression - gaussianization    ##############")
    L = [1e-6, 1e-4, 1e-2]
#    evaluation_LR(DTR, LTR, L, 'GAUSSIANIZED_')

    print("############    Support Vector Machine - gaussianization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
#   evaluation_SVM(DTR, LTR, K_arr, C_arr, 'GAUSSIANIZED_')


##    DTR = stats.zscore(DTR, axis=1)
#    plot_features(DTR, LTR, 'ZNORM')

    print("############    MVG - Z Normalization    ##############")
    evaluation_MVG(DTR, LTR, DTE, LTE, 'ZNORM')

    print("############    Logistic Regression - Z Normalization    ##############")
    L = [1e-6, 1e-4, 1e-2]
#    evaluation_LR(DTR, LTR, L, 'ZNORM_')

    print("############    Support Vector Machine - Z Normalization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
#   evaluation_SVM(DTR, LTR, K_arr, C_arr, 'ZNORM')
