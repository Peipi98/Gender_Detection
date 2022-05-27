from mlFunc import *
from validators import *
from classifiers import *

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    plot(DTR, LTR)
    DTE, LTE = load("Test.txt")
    # plot_hist(D, L)

    # We're starting with Multivariate Gaussian Classifier
    _, LPred2 = MGC(DTE, DTR, LTR)
    _, LP2n = naive_MGC(DTE, DTR, LTR)
    _, LP2t = tied_cov_GC(DTE, DTR, LTR)
    _, LP2nt = tied_cov_naive_GC(DTE, DTR, LTR)
    # logMGC accuracy
    log_acc, log_err = test(LTE, LPred2)
    log_acc_n, log_err_n = test(LTE, LP2n)
    log_acc_t, log_err_t = test(LTE, LP2t)
    log_acc_nt, log_err_nt = test(LTE, LP2nt)
    
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
    #kfold_cross(MGC, DTR, LTR, 10)

    # DA CHIEDERE
    # Notiamo che i risultati di leave-one-out sono rispettivamente
    # più bassi rispetto ai precedenti non naive, ma più alti dei naive.
    # 0.9753333333333334 
    # 0.7031666666666667
    # 0.9755
    # 0.7048333333333333

    # Notiamo che le features sono molto correlate tra loro,
    # quindi non possiamo fare l'assunzione di indipendenza di Naive Bayes


    #DTR = LDA(DTR, LTR, 2)



    # _, LPred2 = MGC(DTE, DTR, LTR)
    #print(test(LTE, LPred2))

    print('Linear regression:')
    lamb = [0.0, 1e-6, 1e-3, 0.1, 1.0, 3.0]
    
    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        print( str(l)+ "\t\t" + str(acc_LR) + "\t" + str(round(err_LR*100, 3))+"%")


    print('Linear regression + PCA(12 -> 11):')
    #todo we still have to find the best 'm' by using validation
    m = 8
    P = PCA(DTR, LTR, m)
    DTR = numpy.dot(P.T, DTR)
    plot(DTR, LTR)
    DTE = numpy.dot(P.T, DTE)
    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        print( str(l)+ "\t\t" + str(acc_LR) + "\t" + str(round(err_LR*100, 3))+"%")

    print('Linear regression + LDA(binary case -> d=1 direction):')
    # todo we still have to find the best 'm' by using validation
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")

    W = LDA(DTR, LTR, 1)
    DTR = numpy.dot(W.T, DTR)
    DTE = numpy.dot(W.T, DTE)

    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        print(str(l) + "\t\t" + str(acc_LR) + "\t" + str(round(err_LR * 100, 3)) + "%")


    print('Linear regression + PCA(12 -> 11) + LDA(binary case -> d=1 direction):')
    # todo we still have to find the best 'm' by using validation
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")

    m = 8
    P = PCA(DTR, LTR, m)
    DTR = numpy.dot(P.T, DTR)
    DTE = numpy.dot(P.T, DTE)

    W = LDA(DTR, LTR, 1)
    DTR = numpy.dot(W.T, DTR)
    DTE = numpy.dot(W.T, DTE)


    for l in lamb:
        LPred, _J = linear_reg(DTR, LTR, DTE, l)
        acc_LR, err_LR = test(LTE, LPred)
        print(str(l) + "\t\t" + str(acc_LR) + "\t" + str(round(err_LR * 100, 3)) + "%")
