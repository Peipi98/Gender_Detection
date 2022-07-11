from validation.validation_LR import validation_LR
from validation.validation_weighted_LR import validation_weighted_LR
from validation.validation_quad_LR import validation_quad_LR
from validation.validation_MVG import validation_MVG
from validation.validation_SVM import validation_SVM
from validation.validation_SVM_RFB import validation_SVM_RFB
from validation.validation_SVM_polynomial import validation_SVM_polynomial
from plot_features import plot_features
from validators import *
import scipy.stats as stats


if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    #    plot_features(DTR, LTR, 'RAW_')

    # RAW data

    print("############    MVG - RAW   ##############")
#    validation_MVG(DTR, LTR, DTE, LTE, 'RAW_')

    print("############    Logistic Regression - RAW    ##############")
    L = [1e-6, 1e-4, 1e-2, 1.0]

#    validation_LR(DTR, LTR, L, 'RAW_')
    L = [1e-4, 1e-2, 1e-1, 1.0]
#    validation_weighted_LR(DTR, LTR, L, 'RAW_')
    #validation_quad_LR(DTR, LTR, L, 'RAW_')

    print("############    Support Vector Machine - RAW    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    #validation_SVM(DTR, LTR, K_arr, C_arr, 'RAW_')
    CON_array = [1000]
    K_arr = [1., 10.]

    #validation_SVM_polynomial(DTR, LTR, K_arr, 1.0, 'RAW_', CON_array, False)
    K_arr = [0.1, 1., 10.]
    C_arr = [1., 10.]
    gamma_Arr = [0.001]
    #validation_SVM_RFB(DTR, LTR, K_arr, gamma_Arr, 'RAW_', PCA_Flag=False)

    # Gaussianization

##    DTR = gaussianize_features(DTR, DTR)
    #    plot_features(DTR, LTR, 'GAUSSIANIZED_')

    print("############    MVG - gaussianization    ##############")
#    validation_MVG(DTR, LTR, DTE, LTE, 'GAUSSIANIZED_')

    print("############    Logistic Regression - gaussianization    ##############")
    L = [1e-6, 1e-4, 1e-2]
#    validation_LR(DTR, LTR, L, 'GAUSSIANIZED_')

    print("############    Support Vector Machine - gaussianization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
#   validation_SVM(DTR, LTR, K_arr, C_arr, 'GAUSSIANIZED_')


    DTR = stats.zscore(DTR, axis=1)
#    plot_features(DTR, LTR, 'ZNORM')

    print("############    MVG - Z Normalization    ##############")
    #validation_MVG(DTR, LTR, DTE, LTE, 'ZNORM')

    print("############    Logistic Regression - Z Normalization    ##############")
    L = [1e-6, 1e-4, 1e-2]
#    validation_LR(DTR, LTR, L, 'ZNORM_')
    L = [1e-4, 1e-2, 1e-1, 1.0]
    #validation_weighted_LR(DTR, LTR, L, 'RAW_', PCA_Flag=False)

    print("############    Support Vector Machine - Z Normalization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
#   validation_SVM(DTR, LTR, K_arr, C_arr, 'ZNORM')
