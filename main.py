#from functions.calibrationFunc import calibrate_SVM
'''
+---------------------------+
|                           |
|         VALIDATION        |
|                           |
+---------------------------+
'''
from validation.validation_LR import validation_LR
from validation.validation_weighted_LR import validation_weighted_LR
from validation.validation_quad_LR import validation_quad_LR
from validation.validation_MVG import validation_MVG
from validation.validation_SVM import validation_SVM
from validation.validation_SVM_RFB import validation_SVM_RFB
from validation.validation_SVM_polynomial import validation_SVM_polynomial
from validation.validation_GMM import validation_GMM_ncomp

'''
+---------------------------+
|                           |
|         EVALUATION        |        
|                           |
+---------------------------+
'''
from evaluators.evaluation_MVG import evaluation_MVG
from evaluators.evaluation_GMM import evaluation_GMM_ncomp

#from plot_features import plot_features
from validators import *
import scipy.stats as stats


if __name__ == "__main__":

    DTR, LTR = load("Train.txt")
    DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    DTE, LTE = randomize(DTE, LTE)
    #    plot_features(DTR, LTR, 'RAW_')
    '''
    # RAW data

    print("############    MVG - RAW   ##############")
    validation_MVG(DTR, LTR 'RAW_')

    print("############    Logistic Regression - RAW    ##############")
    L = [1e-6, 1e-4, 1e-2, 1.0] #1e-6

    validation_LR(DTR, LTR, L, 'RAW_')
    L = [1e-4, 1e-2, 1e-1, 1.0]
    validation_weighted_LR(DTR, LTR, L, 'RAW_', PCA_Flag=False)
    validation_quad_LR(DTR, LTR, L, 'RAW_')

    print("############    Support Vector Machine - RAW    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    K_arr = [1.0] #
    C_arr = [1.0] #
    validation_SVM(DTR, LTR, K_arr, C_arr, 'RAW_')
    #calibrate_SVM(DTR, LTR, 'RAW_CALIBRATED_')
    CON_array = [1000]
    K_arr = [1., 10.]

    validation_SVM_polynomial(DTR, LTR, K_arr, 1.0, 'RAW_', CON_array, False)
    K_arr = [0.1, 1., 10.]
    C_arr = [1., 10.]
    gamma_Arr = [0.001]
    validation_SVM_RFB(DTR, LTR, K_arr, gamma_Arr, 'RAW_', PCA_Flag=False)
    
    print("############    Gaussian Mixture Model - RAW & Gaussianization    ##############")
    DTR_gauss = gaussianize_features(DTR, DTR)
    validation_GMM_ncomp(DTR, DTR_gauss, LTR, pi, n)

    # Gaussianization

    DTR = gaussianize_features(DTR, DTR)
    #    plot_features(DTR, LTR, 'GAUSSIANIZED_')

    print("############    MVG - gaussianization    ##############")
    '''
    validation_MVG(DTR, LTR, 'GAUSSIANIZED_', Gauss_flag=True)
    '''
    print("############    Logistic Regression - gaussianization    ##############")
    L = [1e-6, 1e-4, 1e-2]
    validation_LR(DTR, LTR, L, 'GAUSSIANIZED_')

    print("############    Support Vector Machine - gaussianization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    validation_SVM(DTR, LTR, K_arr, C_arr, 'GAUSSIANIZED_')


    DTR = stats.zscore(DTR, axis=1)
#    plot_features(DTR, LTR, 'ZNORM')

    print("############    MVG - Z Normalization    ##############")
    validation_MVG(DTR, LTR, DTE, LTE, 'ZNORM')

    print("############    Logistic Regression - Z Normalization    ##############")
    L = [1e-6, 1e-4, 1e-2]
    validation_LR(DTR, LTR, L, 'ZNORM_')
    L = [1e-4, 1e-2, 1e-1, 1.0]
    validation_weighted_LR(DTR, LTR, L, 'RAW_', PCA_Flag=False)

    print("############    Support Vector Machine - Z Normalization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    validation_SVM(DTR, LTR, K_arr, C_arr, 'ZNORM')
    '''

    '''
    +---------------------------+
    |                           |
    |         EVALUATION        |        
    |                           |
    +---------------------------+
    '''

    DTR, LTR = load("Train.txt")
    DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    DTE, LTE = randomize(DTE, LTE)


    print("############    MVG - RAW   ##############")
    #validation_MVG(DTR, LTR, 'RAW_')
    evaluation_MVG(DTR, LTR, DTE, LTE, 'RAW_', False)
    '''
    print("############    Logistic Regression - RAW    ##############")
    L = [1e-6]  # 1e-6

    validation_LR(DTR, LTR, L, 'RAW_')
    L = [1e-4]
    validation_weighted_LR(DTR, LTR, L, 'RAW_', PCA_Flag=False)
    validation_quad_LR(DTR, LTR, L, 'RAW_')

    print("############    Support Vector Machine - RAW    ##############")
    K_arr = [1.0]
    C_arr = [1.0]
    validation_SVM(DTR, LTR, K_arr, C_arr, 'RAW_')
    # calibrate_SVM(DTR, LTR, 'RAW_CALIBRATED_')
    CON_array = [1000]
    K_arr = [1.]

    validation_SVM_polynomial(DTR, LTR, K_arr, 1.0, 'RAW_', CON_array, False)
    K_arr = [1.]
    C_arr = [10.]
    gamma_Arr = [0.001]
    validation_SVM_RFB(DTR, LTR, K_arr, gamma_Arr, 'RAW_', PCA_Flag=False)
    '''
    print("############    Gaussian Mixture Model - RAW    ##############")
    evaluation_GMM_ncomp("RAW", DTR, LTR, DTE, LTE, 0.5, 2)
    evaluation_GMM_ncomp("RAW", DTR, LTR, DTE, LTE, 0.1, 2)
    evaluation_GMM_ncomp("RAW", DTR, LTR, DTE, LTE, 0.9, 2)

    # Gaussianization

    DTR = gaussianize_features(DTR, DTR)
    DTE = gaussianize_features(DTR, DTE)
    #    plot_features(DTR, LTR, 'GAUSSIANIZED_')

    print("############    MVG - gaussianization    ##############")
    evaluation_MVG(DTR, LTR, DTE, LTE, 'GAUSSIANIZED_')
    '''
    print("############    Logistic Regression - gaussianization    ##############")
    L = [1e-6, 1e-4, 1e-2]
    validation_LR(DTR, LTR, L, evaluation_MVG(DTR, LTR, DTE, LTE, 'RAW_'))
    
    print("############    Support Vector Machine - gaussianization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    validation_SVM(DTR, LTR, K_arr, C_arr, 'GAUSSIANIZED_')
    '''
    print("############    Gaussian Mixture Model - gaussianization    ##############")
    evaluation_GMM_ncomp("gauss.", DTR,LTR, DTE, LTE, 0.5, 2)
    evaluation_GMM_ncomp("gauss.", DTR, LTR, DTE, LTE, 0.1, 2)
    evaluation_GMM_ncomp("gauss.", DTR, LTR, DTE, LTE, 0.9, 2)
    


    '''
    DTR = stats.zscore(DTR, axis=1)
    #    plot_features(DTR, LTR, 'ZNORM')

    print("############    MVG - Z Normalization    ##############")
    evaluation_MVG(DTR, LTR, DTE, LTE, 'GAUSSIANIZED_')

    print("############    Logistic Regression - Z Normalization    ##############")
    L = [1e-6, 1e-4, 1e-2]
    validation_LR(DTR, LTR, L, 'ZNORM_')
    L = [1e-4, 1e-2, 1e-1, 1.0]
    validation_weighted_LR(DTR, LTR, L, 'RAW_', PCA_Flag=False)

    print("############    Support Vector Machine - Z Normalization    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    validation_SVM(DTR, LTR, K_arr, C_arr, 'ZNORM')


'''
