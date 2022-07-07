import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
import scipy.optimize as opt
from prettytable import PrettyTable


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        try:
            for line in f:
                attrs = line.replace(" ", "").split(',')[0:12]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = int(name)
                DList.append(attrs)
                labelsList.append(label)
        except:
            pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def emprical_mean(D):
    return mcol(D.mean(1))


def empirical_covariance(D, mu):
    n = numpy.shape(D)[1]
    DC = D - mcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    return C


def plot_PCA_result(P, D, L, m, filename, LDA_flag):
    plt.figure()
    DP = numpy.dot(P.T, D)
    hlabels = {
        0: "male",
        1: "female"
    }
    if m == 2:
        for i in range(2):
            # I have to invert the sign of the second eigenvector to flip the image
            plt.scatter(DP[:, L == i][0], -DP[:, L == i][1], label=hlabels.get(i), s=10)
            plt.legend()
            plt.tight_layout()
        if LDA_flag is True:
            DTR = numpy.dot(P.T, -D)
            W = LDA(DTR, L, 1) * 100

            plt.quiver(W[0] * -5, W[1] * -5, W[0] * 40, W[1] * 40, units='xy', scale=1, color='g')
            plt.xlim(-65, 65)
            plt.ylim(-25, 25)
        plt.savefig('./images/' + filename + '.png')
        plt.show()
    if m == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        for i in range(2):
            x_vals = DP[:, L == i][0]
            y_vals = -DP[:, L == i][1]
            z_vals = DP[:, L == i][2]

            # I have to invert the sign of the second eigenvector to flip the image
            ax.scatter(x_vals, y_vals, z_vals, label=hlabels.get(i), s=10)
            plt.legend()
            plt.tight_layout()

        if LDA_flag is True:
            DTR = numpy.dot(P.T, -D)
            W = LDA(DTR, L, 1) * 100
            W = W.ravel()
            x = numpy.array([W[0] * -3, W[0] * 3])
            y = numpy.array([W[1] * -3, W[1] * 3])
            z = numpy.array([W[2] * -3, W[2] * 3])
            ax.plot3D(x, y, z)
            ax.view_init(270, 270)

        plt.savefig('./images/' + filename + '.png')
        plt.show()


def PCA(D, L, m=2, filename=None, LDA_flag=False):
    n = numpy.shape(D)[1]
    mu = D.mean(1)
    DC = D - mcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    USVD, s, _ = numpy.linalg.svd(C)
    P = USVD[:, 0:m]

    if filename is not None:
        plot_PCA_result(P, D, L, m, filename, LDA_flag)

    return P


def LDA(D, L, d=1, m=2):
    N = numpy.shape(D)[1]
    mu = D.mean(1)

    tot = 0
    for i in range(m):
        nc = D[:, L == i].shape[1]
        muc = D[:, L == i].mean(1)
        tot += nc * (mcol(muc - mu)).dot(mcol(muc - mu).T)

    SB = 1 / N * tot

    SW = 0
    for i in range(2):
        SW += (L == i).sum() * empirical_covariance(D[:, L == i], emprical_mean(D))

    SW = SW / N

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:d]
    return W


def gaussianize_features(DTR, TO_GAUSS):
    P = []

    for dIdx in range(DTR.shape[0]):
        DT = mcol(TO_GAUSS[dIdx, :])
        X = DTR[dIdx, :] < DT
        R = (X.sum(1) + 1) / (DTR.shape[1] + 2)
        P.append(scipy.stats.norm.ppf(R))
    return numpy.vstack(P)


def plot_histogram(D, L, labels, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    y = D[:, L == 0]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[0])
    y = D[:, L == 1]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[1])
    matplotlib.pyplot.legend()
    plt.savefig('./images/hist' + title + '.png')
    matplotlib.pyplot.show()


def ML_GAU(D):
    m = emprical_mean(D)
    C = empirical_covariance(D, m)
    return m, C


def logpdf_GAU_ND(X, mu, C):
    P = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2 * numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]

    Y = []

    for i in range(X.shape[1]):
        x = X[:, i:i + 1]
        res = const + -0.5 * numpy.dot((x - mu).T, numpy.dot(P, (x - mu)))
        Y.append(res)
    return numpy.array(Y).ravel()


def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()


def likelihood(XND, m_ML, C_ML):
    return numpy.exp(loglikelihood(XND, m_ML, C_ML))


def test(LTE, LPred):
    accuracy = (LTE == LPred).sum() / LTE.size
    error = 1 - accuracy
    return accuracy, error


def logreg_obj_wrap(DTR, LTR, l):
    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        S = numpy.dot(w.T, DTR) + b
        cxe = numpy.logaddexp(0, -S * Z)
        return numpy.linalg.norm(w) ** 2 * l / 2.0 + cxe.mean()

    return logreg_obj


def plot(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    hLabels = {
        0: "male",
        1: "female"
    }
    plt.figure()
    for i in range(2):
        plt.legend(hLabels)
        plt.scatter(D[:, L == i][0], D[:, L == i][1], label=hLabels.get(i))
    plt.show()


def calculate_lbgf(H, DTR, C):
    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=1.0,
        maxiter=100000,
        maxfun=100000,
    )

    return alphaStar, JDual, LDual


def train_SVM_linear(DTR, LTR, C, K=1):
    DTREXT = numpy.vstack([DTR, K * numpy.ones((1, DTR.shape[1]))])
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.dot(DTREXT.T, DTREXT)
    # Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
    # H = numpy.exp(-Dist)
    H = mcol(Z) * mrow(Z) * H

    def JPrimal(w):
        S = numpy.dot(mrow(w), DTREXT)
        loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * numpy.linalg.norm(w) ** 2 + C * loss

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)
    # print(_x),
    # print(_y)

    wStar = numpy.dot(DTREXT, mcol(alphaStar) * mcol(Z))

    # print (JPrimal(wStar))
    # print (JDual(alphaStar)[0])

    def get_duality_gap(alpha, w):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        return JPrimal(w) - (- 0.5 * aHa.ravel() + numpy.dot(mrow(alpha), numpy.ones(alpha.size)))

    return wStar, JPrimal(wStar), JDual(alphaStar)[0], get_duality_gap(alphaStar, wStar);


def train_SVM_polynomial(DTR, LTR, C, K=1, constant=0, degree=2):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = (numpy.dot(DTR.T, DTR) + constant) ** degree + K ** 2
    # Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
    # H = numpy.exp(-Dist)
    H = mcol(Z) * mrow(Z) * H

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

    return alphaStar, JDual(alphaStar)[0]


def train_SVM_RBF(DTR, LTR, C, K=1, gamma=1.):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # kernel function
    kernel = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernel[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = mcol(Z) * mrow(Z) * kernel

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

    return alphaStar, JDual(alphaStar)[0]
