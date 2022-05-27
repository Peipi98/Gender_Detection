from cProfile import label
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
    n = numpy.shape(D)[1]
    mu = D.mean(1)
    DC = D - mcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    USVD, s, _ = numpy.linalg.svd(C)
    P = USVD[:, 0:m]
    DP = numpy.dot(P.T, D)
    hlabels = {
        0: "male",
        1: "female"
    }

    if m == 2:
        for i in range(2):
            # I have to invert the sign of the second eigenvector to flip the image
            plt.scatter(DP[:, L == i][0], -DP[:, L == i][1], label=hlabels.get(i))
            plt.legend()
            plt.tight_layout()
        plt.show()
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


    if d == 1:
        for i in range(m):
            y = numpy.dot(W.T, D[:, L == i])
            matplotlib.pyplot.scatter(y[0], numpy.zeros(y.shape[1]))
        matplotlib.pyplot.show()

    return W

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


def plot(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    hLabels = {
        0: "Male",
        1: "Female"
    }
    plt.figure()
    for i in range(2):
        plt.scatter(D[:, L==i][0],D[:, L==i][1], label = hLabels.get(i) )
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
