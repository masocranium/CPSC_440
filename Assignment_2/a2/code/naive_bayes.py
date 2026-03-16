import numpy as np
from scipy.special import logsumexp

from kmeans import KMeans


class NaiveBayes:
    def __init__(self, prior_alpha=1, prior_beta=1, X=None, y=None):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}

        # num_classes is just a "backup" to pass k in case you don't actually see
        # any from the highest class

        n, self.d = X.shape

        # categorical MLE for y:
        self.y_probs = np.bincount(y) / n
        self.num_classes = k = len(self.y_probs)

        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class VQNB:
    def __init__(self, k, prior_alpha=1, prior_beta=1, X=None, y=None):
        self.k = k
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes Xs are in {0, 1}, ys are in {0, ..., k-1}

        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()
