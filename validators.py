import numpy
from mlFunc import *

def kfold_cross(func, DTR, LTR, k):
    accuracy = []
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(numpy.hstack(Dtr[i + 1:]))
            L.append(numpy.hstack(Ltr[i + 1:]))
        elif i == k - 1:
            D.append(numpy.hstack(Dtr[:i]))
            L.append(numpy.hstack(Ltr[:i]))
        else:
            D.append(numpy.hstack(Dtr[:i]))
            D.append(numpy.hstack(Dtr[i + 1:]))
            L.append(numpy.hstack(Ltr[:i]))
            L.append(numpy.hstack(Ltr[i + 1:]))

        D = numpy.hstack(D)
        L = numpy.hstack(L)

        DTE = Dtr[i]
        LTE = Ltr[i]
        # print(str(DTE) + " " + str(i))
        _, lpred = func(DTE, LTE, D, L)
        acc, _ = test(LTE, lpred)
        accuracy.append(acc)

    return numpy.mean(accuracy)


def holdout_validation(func, D, L, seed=0, trainPerc=0.8):
    nTrain = int(D.shape[1] * trainPerc)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    _, lpred = func(DTE, LTE, DTR, LTR)
    acc, _ = test(LTE, lpred)

    return acc


def leave_one_out(func, DTR, LTR):
    accuracy = []

    for i in range(DTR.shape[1]):
        D = []
        L = []
        D.append(DTR[:, :i])
        D.append(DTR[:, i + 1:])
        D = numpy.hstack(D)

        L.append(LTR[:i])
        L.append(LTR[i + 1:])
        L = numpy.hstack(L)

        DTE = DTR[:, i]
        LTE = LTR[i]
        # print(str(DTE) + " " + str(i))
        _, lpred = func(mcol(DTE), mcol(LTE), D, L)
        acc, _ = test(LTE, lpred)
        accuracy.append(acc)
    return numpy.mean(accuracy)
