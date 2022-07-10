# -*- coding: utf-8 -*-

import numpy
import scipy
from mlFunc import empirical_covariance, mrow, mcol, logpdf_GAU_ND, get_DTRs
from validators import compute_min_DCF


def initialize_GMM(D, n):
    gmm = []

    for i in range(n):
        weights = 1/n
        mu = mcol(numpy.array(D.mean(1)))
        C = numpy.matrix(empirical_covariance(D, mu))

        gmm.append((weights, mu, C))

    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]


def GMM_ll_perSample(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G, N))

    for g in range(G):
        S[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)


def GMM_EM_full(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = numpy.dot(U, mcol(s)*U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        # print(llNew)
    #print(llNew-llOld)
    return gmm


def GMM_EM_diag(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            Sigma = Sigma * numpy.eye(Sigma.shape[0])
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            covNew = numpy.dot(U, mcol(s)*U.T)
            gmmNew.append((w, mu, covNew))
        gmm = gmmNew
        # print(llNew)
    # print(llNew)
    return gmm


def GMM_EM_tied_full(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM tied full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    sigma_array = []
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            Sigma = Sigma * Z
            gmmNew.append((w, mu, Sigma))

        # calculate tied covariance

        sigma_array = []
        for g in range(G):
            sigma_array.append(gmmNew[g][2])
        tiedSigma = numpy.array(sigma_array).sum(axis=0) / N
        U, s, _ = numpy.linalg.svd(tiedSigma)
        s[s < psi] = psi
        covNew = numpy.dot(U, mcol(s)*U.T)
        gmmNew = [(w, mu, covNew) for w, mu, _ in gmmNew]

        gmm = gmmNew
        # print(llNew)
    # print(llNew)

    return gmm


def GMM_EM_tied_diag(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM tied diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    sigma_array = []
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = numpy.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            Sigma = Sigma * numpy.eye(Sigma.shape[0])
            Sigma = Sigma * Z
            gmmNew.append((w, mu, Sigma))

        # calculate tied covariance

        sigma_array = []
        for g in range(G):
            sigma_array.append(gmmNew[g][2])
        tiedSigma = numpy.array(sigma_array).sum(axis=0) / N
        U, s, _ = numpy.linalg.svd(tiedSigma)
        s[s < psi] = psi
        covNew = numpy.dot(U, mcol(s)*U.T)
        gmmNew = [(w, mu, covNew) for w, mu, _ in gmmNew]

        gmm = gmmNew
        # print(llNew)
    # print(llNew)

    return gmm


def GMM_split(gmm, alpha=0.1):
    '''
    Split a GMM into two GMMs
    '''
    size = len(gmm)
    splitted = []
    for i in range(size):
        U, s, V = numpy.linalg.svd(gmm[i][2])
        d = U[:, 0:1] * s[0]**0.5 * alpha
        # (wg/2, mu+d, sigma)
        splitted.append((gmm[i][0]/2, gmm[i][1]+d, gmm[i][2]))
        # (wg/2, mu-d, sigma)
        splitted.append((gmm[i][0]/2, gmm[i][1]-d, gmm[i][2]))
    return splitted


def GMM_LBG(X, iter, alpha=0.1, psi=0.01, type='full'):
    '''
    LBG algorithm for GMM
    iter is the number of iterations, for each iteration it
    split the data in two parts:
    (wg/2, mu+d, sigma) and (wg/2, mu-d, sigma)
    if iter == 0 it returns gmm with only 1 component
    '''
    # start with 1g
    wg = 1.0
    mu = mcol(X.mean(1))
    c = empirical_covariance(X, mu)
    gmm_1 = [(wg, mu, c)]
    # initialize gmm
    # gmm = EM_FN(X, gmm_1, psi)
    if type == 'full':
        gmm = GMM_EM_full(X, gmm_1, psi)
    elif type == 'diag':
        gmm = GMM_EM_diag(X, gmm_1, psi)
    elif type == 'tied_full':
        gmm = GMM_EM_tied_full(X, gmm_1, psi)
    elif type == 'tied_diag':
        gmm = GMM_EM_tied_diag(X, gmm_1, psi)

    # iter == 0 -> 1g
    if iter == 0:
        return gmm

    for i in range(iter):
        gmm = GMM_split(gmm, alpha=alpha)
        # gmm = EM_FN(X, gmm, psi)
        if type == 'full':
            gmm = GMM_EM_full(X, gmm, psi)
        elif type == 'diag':
            gmm = GMM_EM_diag(X, gmm, psi)
        elif type == 'tied_full':
            gmm = GMM_EM_tied_full(X, gmm, psi)
        elif type == 'tied_diag':
            gmm = GMM_EM_tied_diag(X, gmm, psi)
    return gmm


class GMM:
    def __init__(self, DTR, LTR, DTE, LTE, prior_prob_array, iterations=2, alpha=0.1, psi=0.01, typeOfGmm="full"):
        # initialization of the attributes
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.iterations = iterations if iterations == 0 else int(numpy.log2(iterations))
        self.type = typeOfGmm
        self.alpha = alpha
        self.psi = psi

        self.mu_array = []
        self.cov_array = []
        self.prior_prob_array = prior_prob_array

        self.gmm_array = []
        self.SPost = []
        self.llrs = 0.
        self.predicted_labels = []
        self.accuracy = 0.
        self.error = 0.

        self.dcf = 0.

    def train(self):

        DTR_array = get_DTRs(self.DTR, self.LTR, self.LTR.max() + 1)

        for DTRi in DTR_array:
            mu_i = numpy.mean(DTRi, axis=1)
            mu_i = mu_i.reshape((mu_i.shape[0], 1))
            cov_i = 1/DTRi.shape[1] * numpy.dot(DTRi-mu_i, (DTRi-mu_i).T)

            self.mu_array.append(mu_i)
            self.cov_array.append(cov_i)

            self.gmm_array.append(
                GMM_LBG(DTRi, self.iterations, self.alpha, self.psi, self.type))

    def test(self):
        logS = numpy.zeros((2, self.DTE.shape[1]))

        for i, gmm_i in enumerate(self.gmm_array):
            logS[i, :] = GMM_ll_perSample(self.DTE, gmm_i)

        logSJoint = numpy.vstack((logS))
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal

        self.SPost = numpy.exp(logSPost)

        self.predicted_labels = numpy.argmax(self.SPost, axis=0)

        self.accuracy = numpy.sum(
            self.LTE == self.predicted_labels) / len(self.LTE)
        self.error = 1 - self.accuracy

        self.llrs = logSPost[1, :] - logSPost[0, :]

# 	def compute_dcf(self, threshold = None):
# 		confusion_matrix = compute_confusion_matrix_binary(self.LTE, self.llrs, self.prior_prob_array[1],1,1, threshold)
# 		return compute_normalized_dcf_binary(confusion_matrix, self.prior_prob_array[1], 1, 1)


    def compute_min_dcf(self):
        # compute_min_DCF(scores, labels, pi, Cfn, Cfp):

        return compute_min_DCF(self.llrs, self.LTE, self.prior_prob_array[1], 1, 1)
