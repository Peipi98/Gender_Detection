import numpy
from mlFunc import *

def MGC(DTE, DTR, LTR):
    h = {}

    for i in range(2):
        mu, C = ML_GAU(DTR[:, LTR == i])
        h[i] = (mu, C)

    SJoint = numpy.zeros((2, DTE.shape[1]))
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    dens = numpy.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu, C = h[label]
        dens[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, C).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + numpy.log(classPriors[label])


    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)
    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, numpy.log(dens[1]/dens[0])

def naive_MGC(DTE, DTR, LTR):
    h = {}

    for i in range(2):
        mu, C = ML_GAU(DTR[:, LTR == i])
        C = C * numpy.identity(C.shape[0])
        h[i] = (mu, C)

    SJoint = numpy.zeros((2, DTE.shape[1]))
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    dens = numpy.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu, C = h[label]
        dens[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, C).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + numpy.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, numpy.log(dens[1]/dens[0])

def tied_cov_GC(DTE, DTR, LTR):
    h = {}
    Ctot = 0
    for i in range(2):
        mu, C = ML_GAU(DTR[:, LTR == i])
        Ctot += DTR[:, LTR == i].shape[1] * C
        h[i] = (mu)

    Ctot = Ctot / DTR.shape[1]

    SJoint = numpy.zeros((2, DTE.shape[1]))
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    dens = numpy.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu = h[label]
        dens[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, Ctot).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, Ctot).ravel() + numpy.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, numpy.log(dens[1]/dens[0])


def tied_cov_naive_GC(DTE, DTR, LTR):
    h = {}
    Ctot = 0
    for i in range(2):
        mu, C = ML_GAU(DTR[:, LTR == i])
        Ctot += DTR[:, LTR == i].shape[1] * C
        h[i] = (mu)

    Ctot = Ctot / DTR.shape[1]
    Ctot = Ctot * numpy.identity(Ctot.shape[0])

    SJoint = numpy.zeros((2, DTE.shape[1]))
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    dens = numpy.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu = h[label]
        dens[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, Ctot).ravel())
        SJoint[label, :] = dens[label, :] * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, Ctot).ravel() + numpy.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2, numpy.log(dens[1]/dens[0])


def linear_reg(DTR, LTR, DTE, l):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    _v, _J, _d = opt.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    STE = numpy.dot(_w.T, DTE) + _b
    LP = STE > 0
    return LP, _J

def generative_acc_err(DTE, DTR, LTE, LTR, title):
    _, LPred2 = MGC(DTE, DTR, LTR)
    _, LP2n = naive_MGC(DTE, DTR, LTR)
    _, LP2t = tied_cov_GC(DTE, DTR, LTR)
    _, LP2nt = tied_cov_naive_GC(DTE, DTR, LTR)
    # logMGC accuracy
    log_acc, log_err = test(LTE, LPred2)
    log_acc_n, log_err_n = test(LTE, LP2n)
    log_acc_t, log_err_t = test(LTE, LP2t)
    log_acc_nt, log_err_nt = test(LTE, LP2nt)

    table = PrettyTable(["", "Accuracy %", "Error "])
    table.title = title
    table.add_row(["MGC", round(log_acc*100, 3), round(log_err*100, 3)])
    table.add_row(["Naive MGC", round(log_acc_n*100, 3), round(log_err_n*100, 3)])
    table.add_row(["Tied GC", round(log_acc_t*100, 3), round(log_err_t*100, 3)])
    table.add_row(["Naive Tied GC", round(log_acc_nt*100, 3), round(log_err_nt*100, 3)])
    print(table)

def score_SVM_linear(DTR, LTR, DTE, LTE, _K, _C):
    for K in _K:
        for C in _C:
            wStar, primal, dual, gap = train_SVM_linear(DTR, LTR, C=C, K=K)
            DTEEXT = numpy.vstack([DTE, K*numpy.ones((1, DTE.shape[1]))])
            score = numpy.dot(wStar.T, DTEEXT)
            errorRate = (1 - numpy.sum( (score > 0) == LTE) / len(LTE))*100
            print("K = %d, C = %.1f, primal loss = %e, dual loss = %e, duality gap = %e, errorRate = %.1f" % (K, C, primal, dual, gap, errorRate))
        
def score_SVM_poly(DTR, LTR, DTE, LTE, _K, _d):
    for constant in [0,1]:
        for degree in _d:
            for K in _K:
                
                aStar, loss = train_SVM_polynomial(DTR, LTR, C=1.0, constant=constant, degree=degree, K=K)
                
                
                kernel = (numpy.dot(DTR.T, DTE)+constant)**degree + K*K
                score = numpy.sum( numpy.dot(aStar * mrow(Z), kernel), axis=0 )
                
                errorRate = (1 - numpy.sum( (score > 0) == LTE) / len(LTE))*100
                print("K = %d, constant = %d, loss = %e, error =  %.1f" % (K, constant, loss, errorRate))

def score_SVM_RBF(DTR, LTR, DTE, LTE, _K, _gamma):
    for K in _K:
        for gamma in _gamma:
            aStar, loss = train_SVM_RBF(DTR, LTR, C=1.0, K=K, gamma=gamma)
            
            kern = numpy.zeros((DTR.shape[1], DTE.shape[1]))
            for i in range(DTR.shape[1]):
                for j in range(DTE.shape[1]):
                    kern[i,j] = numpy.exp(-gamma*(numpy.linalg.norm(DTR[:,i]-DTE[:,j])**2))+ K*K
            
            score = numpy.sum( numpy.dot(aStar * mrow(Z), kern), axis=0 )
            
            errorRate = (1 - numpy.sum( (score > 0) == LTE) / len(LTE))*100
            print("K = %d, gamma = %.1f, loss = %e, error =  %.1f" % (K, gamma, loss, errorRate))

        