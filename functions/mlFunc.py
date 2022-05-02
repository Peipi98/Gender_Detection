import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets

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
    

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris() ['target']
    return D, L
    
    
def mu(D):
    return mcol(D.mean(1))

def covariance(D, mu):
    n = numpy.shape(D)[1]
    DC = D - mcol(mu)
    C = 1/n * numpy.dot(DC, numpy.transpose(DC))
    return C

def PCA(D, L):

# 1. compute covariance matrix
    n = numpy.shape(D)[1]
    # mu = dataset mean, calculated by axis = 1 (columns mean)
    # the result is an array of means for each column
    mu = D.mean(1)

    # remove the mean from all points of the data matrix D,
    # so I can center the data
    DC = D - mcol(mu)

    #calculate covariance matrix with DataCentered matrix
    C = 1/n * numpy.dot(DC, numpy.transpose(DC))

    #Calculate eigenvectors and eigenvalues of C with singular value decomposition
    # That's why C is semi-definite positive, so we can get the sorted eigenvectors
    # from the svd: C=U*(Sigma)*V(^transposed)
    # svd() returns sorted eigenvalues from smallest to largest,
    # and the corresponding eigenvectors
    USVD, s, _ = numpy.linalg.svd(C)

    # m are the leading eigenvectors chosen from the next P matrix
    m = 2
    P = USVD[:, 0:m]

    #apply the projection to the matrix of samples D
    DP = numpy.dot(P.T, D)
    print(DP)
    hlabels = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    for i in range(3):
#I have to invert the sign of the second eigenvector to flip the image
        plt.scatter(DP[:, L==i][0], -DP[:, L==i][1], label = hlabels.get(i))
        plt.legend()
        plt.tight_layout()
    plt.show()

def ML_GAU(D):
    m = mu(D)
    C= covariance(D, m)
    return m, C

def logpdf_GAU_ND(X, mu, C):
    P  = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]

    Y = []

    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * numpy.dot((x-mu).T, numpy.dot(P, (x-mu)))
        Y.append(res)
    return numpy.array(Y).ravel()

def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()

def likelihood(XND, m_ML, C_ML):
    return numpy.exp(loglikelihood(XND, m_ML, C_ML))

def plot():
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000) 
    m = numpy.ones((1,1)) * 1.0
    C = numpy.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m, C))) 
    plt.show()

def plot_hist_exp(X1D, m_ML, C_ML):
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(mrow(XPlot), m_ML, C_ML)))
    plt.show()