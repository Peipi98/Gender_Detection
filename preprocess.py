import scipy.stats
from mlFunc import *


def gaussianize_features(D, DE):
    P = []

    for dIdx in range(8):
        DT = mcol(DE[dIdx, :])
        mean = numpy.mean(D[dIdx, :])
        var = numpy.var(D[dIdx, :])
        X = D[dIdx, :] < DT
        R = (X.sum(1) + 1) / (D.shape[1] + 2)
        P.append(scipy.stats.norm.ppf(R))

    return numpy.vstack(P)
