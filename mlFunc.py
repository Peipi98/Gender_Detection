import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
import scipy.optimize as opt


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


def PCA(D, L, m=2):
    # 1. compute covariance matrix
    n = numpy.shape(D)[1]
    # mu = dataset mean, calculated by axis = 1 (columns mean)
    # the result is an array of means for each column
    mu = D.mean(1)

    # remove the mean from all points of the data matrix D,
    # so I can center the data
    DC = D - mcol(mu)

    # calculate covariance matrix with DataCentered matrix
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))

    # Calculate eigenvectors and eigenvalues of C with singular value decomposition
    # That's why C is semi-definite positive, so we can get the sorted eigenvectors
    # from the svd: C=U*(Sigma)*V(^transposed)
    # svd() returns sorted eigenvalues from smallest to largest,
    # and the corresponding eigenvectors
    USVD, s, _ = numpy.linalg.svd(C)

    # m are the leading eigenvectors chosen from the next P matrix
    P = USVD[:, 0:m]

    # apply the projection to the matrix of samples D
    DP = numpy.dot(P.T, D)
    print(DP)
    hlabels = {
        0: "male",
        1: "female"
    }

    for i in range(2):
        # I have to invert the sign of the second eigenvector to flip the image
        plt.scatter(DP[:, L == i][0], -DP[:, L == i][1], label=hlabels.get(i))
        plt.legend()
        plt.tight_layout()
    plt.show()

def LDA(D, L, DTE, m=1):
    N = numpy.shape(D)[1]
    mu = D.mean(1)  # mu -> sumOfRow / 150 -> the result of mean is a 1-D array instead of a column vector.


    tot = 0
    for i in range(2):
        nc = D[:, L == i].shape[1]
        muc = D[:, L == i].mean(1)
        tot += nc * (mcol(muc - mu)).dot(mcol(muc - mu).T)

    SB = 1 / N * tot

    SW = 0
    for i in range(2):
        SW += (L == i).sum() * empirical_covariance(D[:, L == i], emprical_mean(D))

    SW = SW / N

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    Y = []
    for i in range(2):
        y = numpy.dot(W.T, D[:, L == i])
        matplotlib.pyplot.scatter(y[0], y[1])
        Y.append(y)

    matplotlib.pyplot.show()

    return numpy.hstack(Y), DTE

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
    Z = LTR * 2.0 -1.0
    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        S = numpy.dot(w.T, DTR) + b
        cxe = numpy.logaddexp(0, -S*Z)
        return numpy.linalg.norm(w)**2 * l/2.0 + cxe.mean()
    return logreg_obj

# def to_be_transfered_into_main(DTR, LTR, DTE, LTE, lamb):
#     for l in lamb:
        # logreg_obj = logreg_obj_wrap(DTR, LTR, l)
        # _v, _J, _d = opt.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
        # _w = _v[0:DTR.shape[0]]
        # _b = _v[-1]
        # STE = numpy.dot(_w.T, DTE) + _b
        # LP = STE > 0
#         ER = 1 - numpy.array(LP == LTE).mean()
#         print(l, round(_J, 3), str(100*round(ER, 3))+'%')


def plot():
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1, 1)) * 1.0
    C = numpy.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m, C)))
    plt.show()


def plot_hist_exp(X1D, m_ML, C_ML):
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m_ML, C_ML)))
    plt.show()


# PLOTTING FUNCTIONS
def plot_hist(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for dIdx in range(12):
        plt.figure()

        plt.hist(D0[dIdx, :], bins=10, density=True, alpha=0.4, label='male')
        plt.hist(D1[dIdx, :], bins=10, density=True, alpha=0.4, label='female')

        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()
