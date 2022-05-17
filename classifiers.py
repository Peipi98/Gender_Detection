import numpy
from mlFunc import *

def MGC(DTE, DTR, LTR):
    h = {}

    for i in range(2):
        mu, C = ML_GAU(DTR[:, LTR == i])
        h[i] = (mu, C)

    SJoint = numpy.zeros((2, DTE.shape[1]))
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu, C = h[label]
        SJoint[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, C).ravel()) * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + numpy.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2

def naive_MGC(DTE, DTR, LTR):
    h = {}

    for i in range(2):
        mu, C = ML_GAU(DTR[:, LTR == i])
        C = C * numpy.identity(C.shape[0])
        h[i] = (mu, C)

    SJoint = numpy.zeros((2, DTE.shape[1]))
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu, C = h[label]
        SJoint[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, C).ravel()) * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + numpy.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2

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
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu = h[label]

        SJoint[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, Ctot).ravel()) * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, Ctot).ravel() + numpy.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2


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
    classPriors = [0.5, 0.5]

    for label in range(2):
        mu = h[label]

        SJoint[label, :] = numpy.exp(logpdf_GAU_ND(DTE, mu, Ctot).ravel()) * classPriors[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTE, mu, Ctot).ravel() + numpy.log(classPriors[label])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / mrow(SMarginal)
    logPost = logSJoint - mrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.argmax(0)
    LPred2 = Post2.argmax(0)
    return LPred1, LPred2


def linear_reg(DTR, LTR, DTE, l):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    _v, _J, _d = opt.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    STE = numpy.dot(_w.T, DTE) + _b
    LP = STE > 0
    return LP, _J