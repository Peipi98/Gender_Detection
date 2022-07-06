from mlFunc import *
from classifiers import *
from validators import *
from prettytable import PrettyTable

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = gaussianize_features(DTR, DTR)


    DTE = gaussianize_features(DTR, DTE)
    # plot(DTR, LTR)
    # We're starting with Multivariate Gaussian Classifier
    """     
    _, LPred2 = MGC(DTE, DTR, LTR)
    _, LP2n = naive_MGC(DTE, DTR, LTR)
    _, LP2t = tied_cov_GC(DTE, DTR, LTR)
    _, LP2nt = tied_cov_naive_GC(DTE, DTR, LTR)
    # logMGC accuracy
    log_acc, log_err = test(LTE, LPred2)
    log_acc_n, log_err_n = test(LTE, LP2n)
    log_acc_t, log_err_t = test(LTE, LP2t)
    log_acc_nt, log_err_nt = test(LTE, LP2nt) 
    """

    # # GENERATIVE MODELS
    # ## RAW
    # generative_acc_err(DTE, DTR, LTE, LTR, "RAW")
    #
    # ## PCA
    # m = 8
    # P = PCA(DTR, LTR, m)
    # DTR_PCA = numpy.dot(P.T, DTR)
    # DTE_PCA = numpy.dot(P.T, DTE)
    # plot(DTR_PCA, LTR)
    # generative_acc_err(DTE_PCA, DTR_PCA, LTE, LTR, "PCA")
    #
    # ## LDA
    # W = LDA(DTR, LTR, 1)
    # DTR_LDA = numpy.dot(W.T, DTR)
    # DTE_LDA = numpy.dot(W.T, DTE)
    # generative_acc_err(DTE_LDA, DTR_LDA, LTE, LTR, "LDA")
    #
    # ## PCA + LDA
    # W = LDA(DTR_PCA, LTR, 1)
    # DTR_LDA = numpy.dot(W.T, DTR_PCA)
    # DTE_LDA = numpy.dot(W.T, DTE_PCA)
    # generative_acc_err(DTE_LDA, DTR_LDA, LTE, LTR, "PCA + LDA")

    # print(str(round(log_err*100, 3)) + "%")
    # print(str(round(log_err_n*100, 3)) + "%")
    # print(str(round(log_err_t*100, 3)) + "%")
    # print(str(round(log_err_nt*100, 3)) + "%")

    # print(holdout_validation(MGC, DTR, LTR))
    # print(holdout_validation(naive_MGC, DTR, LTR))
    # print(holdout_validation(tied_cov_GC, DTR, LTR))
    # print(holdout_validation(tied_cov_naive_GC, DTR, LTR))
    # 0.9683333333333334
    # 0.71
    # 0.9675
    # 0.7125

    # print(leave_one_out(MGC, DTR, LTR))
    # print(leave_one_out(naive_MGC, DTR, LTR))
    # print(leave_one_out(tied_cov_GC, DTR, LTR))
    # print(leave_one_out(tied_cov_naive_GC, DTR, LTR))
    # kfold_cross(MGC, DTR, LTR, 10)

    # DA CHIEDERE
    # Notiamo che i risultati di leave-one-out sono rispettivamente
    # più bassi rispetto ai precedenti non naive, ma più alti dei naive.
    # 0.9753333333333334
    # 0.7031666666666667
    # 0.9755
    # 0.7048333333333333

    # Notiamo che le features sono molto correlate tra loro,
    # quindi non possiamo fare l'assunzione di indipendenza di Naive Bayes


    # DTR = LDA(DTR, LTR, 2)


    # _, LPred2 = MGC(DTE, DTR, LTR)
    # print(test(LTE, LPred2))

    print('m = 2 -------> ')

    lamb = [0.0, 1e-6, 1e-3, 0.1, 1.0, 3.0]

    linreg_PCA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_PCA.title = 'Linear regression + PCA(12 -> 2)'
    m = 2
    P = PCA(DTR, LTR, m, 'PCA_m=2')
    DTR = numpy.dot(P.T, DTR)
    DTE = numpy.dot(P.T, DTE)
    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_PCA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_PCA)

    linreg_LDA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_LDA.title = 'Linear regression + LDA(binary case -> d=1 direction)'
    # todo we still have to find the best 'm' by using validation
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")


    W = LDA(DTR, LTR, 1)
    DTR = numpy.dot(W.T, DTR)
    DTE = numpy.dot(W.T, DTE)

    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_LDA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_LDA)

    linreg_PCA_LDA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_PCA_LDA.title = 'Linear regression + PCA(12 -> 2) + LDA(binary case -> d=1 direction)'
    # todo we still have to find the best 'm' by using validation
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = gaussianize_features(DTR, DTR)
    DTE = gaussianize_features(DTR, DTE)

    P = PCA(DTR, LTR, m, filename='PCA_m=2 + LDA', LDA_flag=True)
    DTR = numpy.dot(P.T, -DTR)
    DTE = numpy.dot(P.T, -DTE)

    W = LDA(DTR, LTR, 1)
    DTR = numpy.dot(W.T, DTR)
    plot_histogram(DTR, LTR, ['male', 'female'], 'PCA_m=2 + LDA')
    DTE = numpy.dot(W.T, DTE)

    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_PCA_LDA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_PCA_LDA)

    print('m = 3 -------> ')

    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")

    linreg_PCA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_PCA.title = 'Linear regression + PCA(12 -> 3)'
    m = 3
    P = PCA(DTR, LTR, m, 'PCA_m=3')
    DTR = numpy.dot(P.T, DTR)
    DTE = numpy.dot(P.T, DTE)
    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_PCA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_PCA)

    linreg_PCA_LDA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_PCA_LDA.title = 'Linear regression + PCA(12 -> 3) + LDA(binary case -> d=1 direction)'
    # todo we still have to find the best 'm' by using validation
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = gaussianize_features(DTR, DTR)
    DTE = gaussianize_features(DTR, DTE)

    P = PCA(DTR, LTR, m, filename='PCA_m=3 + LDA', LDA_flag=True)
    DTR = numpy.dot(P.T, -DTR)
    DTE = numpy.dot(P.T, -DTE)

    W = LDA(DTR, LTR, 1)
    DTR = numpy.dot(W.T, DTR)
    plot_histogram(DTR, LTR, ['male', 'female'], 'PCA_m=3 + LDA')
    DTE = numpy.dot(W.T, DTE)

    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_PCA_LDA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_PCA_LDA)

    print('m = 8 -------> ')
    linreg = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg.title = 'Linear regression'

    plot_histogram(DTR, LTR, ['male', 'female'], 'No manipulation')
    lamb = [0.0, 1e-6, 1e-3, 0.1, 1.0, 3.0]

    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg)

    linreg_PCA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_PCA.title = 'Linear regression + PCA(12 -> 8)'
    # todo we still have to find the best 'm' by using validation
    m = 8
    P = PCA(DTR, LTR, m)
    DTR = numpy.dot(P.T, DTR)
    plot_histogram(DTR, LTR, ['male', 'female'], 'PCA')
    DTE = numpy.dot(P.T, DTE)
    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_PCA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_PCA)

    linreg_LDA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_LDA.title = 'Linear regression + LDA(binary case -> d=1 direction)'
    # todo we still have to find the best 'm' by using validation
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = gaussianize_features(DTR, DTR)
    DTE = gaussianize_features(DTR, DTE)
    W = LDA(DTR, LTR, 1)
    DTR = numpy.dot(W.T, DTR)
    plot_histogram(DTR, LTR, ['male', 'female'], 'LDA')
    DTE = numpy.dot(W.T, DTE)

    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_LDA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_LDA)

    linreg_PCA_LDA = PrettyTable(["Lambda", "Accuracy %", "Error rate %"])
    linreg_PCA_LDA.title = 'Linear regression + PCA(12 -> 8) + LDA(binary case -> d=1 direction)'
    # todo we still have to find the best 'm' by using validation
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = gaussianize_features(DTR, DTR)
    DTE = gaussianize_features(DTR, DTE)

    m = 8
    P = PCA(DTR, LTR, m)
    DTR = numpy.dot(P.T, -DTR)
    DTE = numpy.dot(P.T, -DTE)

    W = LDA(DTR, LTR, 1)
    DTR = numpy.dot(W.T, DTR)
    plot_histogram(DTR, LTR, ['male', 'female'], 'PCA + LDA')
    DTE = numpy.dot(W.T, DTE)

    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        linreg_PCA_LDA.add_row([l, round(acc_LR * 100, 3), round(err_LR * 100, 3)])
    print(linreg_PCA_LDA)

    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")
    DTR = gaussianize_features(DTR, DTR)
    DTE = gaussianize_features(DTR, DTE)

    _, _, llrs = MVG(DTE, DTR, LTR)
    _, _, llrsn = naive_MVG(DTE, DTR, LTR)
    _, _, llrst = tied_cov_GC(DTE, DTR, LTR)
    _, _, llrsnt = tied_cov_naive_GC(DTE, DTR, LTR)

    plot_ROC(llrs, LTE, 'MVG')
    plot_ROC(llrsn, LTE, 'MVG + Naive')
    plot_ROC(llrst, LTE, 'MVG + Tied')
    plot_ROC(llrsnt, LTE, 'MVG + Naive + Tied')

    # Cfn and Ctp are set to 1
    bayes_error_min_act_plot(llrs, LTE, 'MVG', 0.4)
    bayes_error_min_act_plot(llrsn, LTE, 'MVG + Naive', 1)
    bayes_error_min_act_plot(llrst, LTE, 'MVG + Tied', 0.4)
    bayes_error_min_act_plot(llrsnt, LTE, 'MVG + Naive + Tied', 1)