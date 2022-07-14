from evaluators.evaluation_GMM import evaluation_GMM_ncomp, experimental_GMM
from evaluators.evaluation_LR import evaluation_LR
from evaluators.evaluation_MVG import evaluation_MVG
from evaluators.evaluation_SVM import evaluation_SVM
from evaluators.evaluation_SVM_RFB import evaluation_SVM_RFB
from evaluators.evaluation_SVM_polynomial import evaluation_SVM_polynomial
from evaluators.evaluation_quad_LR import evaluation_quad_LR
from evaluators.evaluation_weighted_LR import evaluation_weighted_LR
from mlFunc import *
from evaluators.compare_best_2 import compute_2best_plots
from plot_features import plot_features
from validation.validation_GMM import validation_GMM_ncomp, validation_GMM_tot
from validation.validation_LR import validation_LR
from validation.validation_MVG import validation_MVG
from validation.validation_SVM import validation_SVM
from validation.validation_SVM_RFB import validation_SVM_RFB
from validation.validation_SVM_polynomial import validation_SVM_polynomial
from validation.validation_compare import compare_2_validation
from validation.validation_quad_LR import validation_quad_LR
from validation.validation_weighted_LR import validation_weighted_LR

def validation(DTR, LTR):
    print("############    MVG    ##############")
    validation_MVG(DTR, LTR, 'RAW_')
    validation_MVG(DTR, LTR, 'GAUSSIANIZED_', Gauss_flag=True)
    validation_MVG(DTR, LTR, 'ZNORM_', zscore=True)

    print("############    Logistic Regression    ##############")
    L = [1e-6, 1e-4, 1e-2, 1.0]
    validation_LR(DTR, LTR, L, 'RAW_', PCA_Flag=True, gauss_Flag=False, zscore_Flag=False)
    validation_LR(DTR, LTR, L, 'GAUSSIANIZED_', PCA_Flag=True, gauss_Flag=True, zscore_Flag=False)
    validation_LR(DTR, LTR, L, 'ZNORMALIZED_', PCA_Flag=True, gauss_Flag=False, zscore_Flag=True)
    print("############    Weighted Logistic Regression    ##############")
    L = [1e-4, 1e-2, 1e-1, 1.0]
    validation_weighted_LR(DTR, LTR, L, 'RAW_', PCA_Flag=True, gauss_Flag=False, zscore_Flag=False)
    validation_weighted_LR(DTR, LTR, L, 'GAUSSIANIZED_', PCA_Flag=True, gauss_Flag=True, zscore_Flag=False)
    validation_weighted_LR(DTR, LTR, L, 'ZNORMALIZED_', PCA_Flag=True, gauss_Flag=False, zscore_Flag=True)

    print("############    Quadratic Logistic Regression    ##############")
    validation_quad_LR(DTR, LTR, L, 'RAW_', PCA_Flag=True, gauss_Flag=False, zscore_Flag=False)

    print("############    Support Vector Machine - Primal    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    validation_SVM(DTR, LTR, K_arr, C_arr, 'RAW_', gauss_Flag=False, zscore_Flag=False)
    validation_SVM(DTR, LTR, K_arr, C_arr, 'GAUSSIANIZED_', gauss_Flag=True, zscore_Flag=False)
    validation_SVM(DTR, LTR, K_arr, C_arr, 'ZNORMALIZED_', gauss_Flag=False, zscore_Flag=True)


    print("############    Support Vector Machine - Dual - Polynomial    ##############")
    K_arr = [1., 10.]
    validation_SVM_polynomial(DTR, LTR, [1.0, 10.0], 1.0, 'RAW_', [1000], PCA_Flag=False, gauss_Flag=False, zscore_Flag=False)

    K_arr = [0.1, 1., 10.]
    C_arr = [1., 10.]

    print("############    Support Vector Machine - Dual - RFB    ##############")
    validation_SVM_RFB(DTR, LTR, K_arr, [0.001], 'RAW_', PCA_Flag=False, gauss_Flag=False, zscore_Flag=False)

    print("############    Gaussian Mixture Model   ##############")

    validation_GMM_tot(DTR, LTR, 0.5)
    validation_GMM_ncomp(DTR, LTR, 0.5, 2)
    validation_GMM_ncomp(DTR, LTR, 0.1, 2)
    validation_GMM_ncomp(DTR, LTR, 0.9, 2)

def evaluation(DTR, LTR, DTE, LTE):
    DTR_GAUSS = gaussianize_features(DTR, DTR)
    DTE_GAUSS = gaussianize_features(DTR, DTE)
    DTR_ZNORM, DTE_ZNORM = znorm(DTR, DTE)

    print("############    MVG   ##############")
    evaluation_MVG(DTR, LTR, DTE, LTE, 'RAW_')
    evaluation_MVG(DTR_GAUSS, LTR, DTE_GAUSS, LTE, 'GAUSSIANIZED_')
    evaluation_MVG(DTR_ZNORM, LTR, DTE_ZNORM, LTE, 'Z-NORM')

    print("############    Logistic Regression    ##############")
    evaluation_LR(DTR, LTR, DTE, LTE, [1e-6], 'EVAL_LR_', PCA_Flag=False)

    evaluation_LR(DTR_GAUSS, LTR, DTE_GAUSS, LTE, [1e-6], 'EVAL_LR_GAUSS_', PCA_Flag=False)
    evaluation_LR(DTR_ZNORM, LTR, DTE_ZNORM, LTE, [1e-6], 'EVAL_LR_ZNORM_', PCA_Flag=False)
    print("############    Weighted Logistic Regression    ##############")
    evaluation_weighted_LR(DTR, LTR, DTE, LTE, [1e-4], 'EVAL_WEIGHTED_LR_', PCA_Flag=False)

    print("############    Quadratic Logistic Regression    ##############")
    evaluation_quad_LR(DTR, LTR, DTE, LTE, [1e-4], 'EVAL_WEIGHTED_LR_', PCA_Flag=False)

    print("############    Support Vector Machine - Primal    ##############")
    evaluation_SVM(DTR, LTR, DTE, LTE, [1.0], [1.0], 'EVAL_SVM_LR', PCA_Flag=False)

    print("############    Support Vector Machine - Dual - Polynomial    ##############")
    evaluation_SVM_polynomial(DTR, LTR, DTE, LTE, [1.0], 1.0, 'EVAL_SVM_POLY', [1000], PCA_Flag=False)

    print("############    Support Vector Machine - Dual - RFB    ##############")
    evaluation_SVM_RFB(DTR, LTR, DTE, LTE, [1.0], [0.001], 'EVAL_SVM_RFB', PCA_Flag=False)

    print("############    Gaussian Mixture Model - RAW    ##############")
    evaluation_GMM_ncomp("RAW", DTR, LTR, DTE, LTE, 0.5, 2)
    evaluation_GMM_ncomp("RAW", DTR, LTR, DTE, LTE, 0.1, 2)
    evaluation_GMM_ncomp("RAW", DTR, LTR, DTE, LTE, 0.9, 2)
    evaluation_GMM_ncomp("gauss.", DTR_GAUSS, LTR, DTE_GAUSS, LTE, 0.5, 2)
    evaluation_GMM_ncomp("gauss.", DTR_GAUSS, LTR, DTE_GAUSS, LTE, 0.1, 2)
    evaluation_GMM_ncomp("gauss.", DTR_GAUSS, LTR, DTE_GAUSS, LTE, 0.9, 2)

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTR, LTR = randomize(DTR, LTR)
    DTE, LTE = load("Test.txt")
    DTE, LTE = randomize(DTE, LTE)
    DTR_GAUSS = gaussianize_features(DTR, DTR)
    plot_features(DTR, LTR, appendToTitle='RAW_')
    plot_features(DTR, LTR, appendToTitle='GAUSSIANIZED_')
    print("############    Validation    ##############")
    validation(DTR, LTR)
    print("############    Evaluation    ##############")
    evaluation(DTR, LTR, DTE, LTE)


    # Warning # Warning # Warning # Warning
    # Warning # Warning # Warning # Warning
    # Warning # Warning # Warning # Warning
    # Warning: the following code has not been cleaned yet. It has been used to generate comparison plots.
    # Creates barcharts for GMM with validation and evaluation
    # experimental_GMM(DTR, LTR, DTE, LTE)
    # Creates bayes error and ROC plots for 2 best models chosen (see inside)
    # compute_2best_plots(DTR, LTR, DTE, LTE)
    # compare_2_validation(DTR, LTR, [1e-4])
