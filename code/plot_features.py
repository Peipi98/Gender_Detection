import sys

import matplotlib.pyplot
import scipy.stats
import seaborn as sns

sys.path.append("./")
from mlFunc import *


def compute_correlation(X, Y):
    x_sum = numpy.sum(X)
    y_sum = numpy.sum(Y)

    x2_sum = numpy.sum(X ** 2)
    y2_sum = numpy.sum(Y ** 2)

    sum_cross_prod = numpy.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = numpy.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr


def plot_correlations(DTR, title, cmap="Greys"):
    corr = numpy.zeros((12, 12))
    for x in range(12):
        for y in range(12):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem

    sns.set()
    heatmap = sns.heatmap(numpy.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    fig = heatmap.get_figure()
    fig.savefig("./images/" + title + ".svg")


def plot_features_histograms(DTR, LTR, _title):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    for i in range(12):
        labels = ["male", "female"]
        title = _title + str(i)
        plt.figure()
        plt.title(title)

        y = DTR[:, LTR == 0][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black',
                 label=labels[0])
        y = DTR[:, LTR == 1][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='blue', edgecolor='black',
                 label=labels[1])
        plt.legend()
        plt.savefig('./images/hist_' + title + '.svg')
        plt.show()


def plot_PCA(DTR, LTR, m, appendToTitle=''):
    PCA(DTR, LTR, m, appendToTitle + 'PCA_m=' + str(m))


def plot_PCA_LDA(DTR, LTR, m, appendToTitle=''):
    P = PCA(DTR, LTR, m, filename=appendToTitle + 'PCA_m=' + str(m) + ' + LDA', LDA_flag=True)
    DTR = numpy.dot(P.T, -DTR)

    W = LDA(DTR, LTR, 1, m)
    DTR = numpy.dot(W.T, DTR)
    plot_histogram(DTR, LTR, ['male', 'female'], 'PCA_m=' + str(m) + ' + LDA')

def plot_features(DTR, LTR, m=2, appendToTitle=''):
    plot_features_histograms(DTR, LTR, appendToTitle + "feature_")
    plot_correlations(DTR, "heatmap_" + appendToTitle)
    plot_correlations(DTR[:, LTR == 0], "heatmap_male_" + appendToTitle, cmap="Blues")
    plot_correlations(DTR[:, LTR == 1], "heatmap_female_" + appendToTitle, cmap="Reds")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    plot_PCA(DTR, LTR, m, appendToTitle)
    plot_PCA(DTR, LTR, m, appendToTitle)

    plot_PCA_LDA(DTR, LTR, m, appendToTitle)
    plot_PCA_LDA(DTR, LTR, m, appendToTitle)
