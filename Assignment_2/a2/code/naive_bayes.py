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

        # Compute theta matrix (k x d) where theta[i, j] = P(X_j = 1 | Y = i)
        self.theta = np.zeros((k, self.d))
        for i in range(k):
            # Get the indices of samples belonging to class i
            class_indices = np.where(y == i)[0]
            # Calculate the number of samples in class i
            n_i = len(class_indices)
            if n_i == 0:
                # If there are no samples for this class, use the prior
                self.theta[i] = (self.prior_alpha) / (self.prior_alpha + self.prior_beta)
            else:
                # Calculate the number of times each feature is 1 in class i
                feature_counts = np.sum(X[class_indices], axis=0)
                # Apply Laplace smoothing
                self.theta[i] = (feature_counts + self.prior_alpha) / (n_i + self.prior_alpha + self.prior_beta)


    def predict(self, X):
        # Compute the log probabilities for each class
        log_probs = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            log_probs[:, i] = np.sum(X * np.log(self.theta[i]) + (1 - X) * np.log(1 - self.theta[i]), axis=1) + np.log(self.y_probs[i])
        # Predict the class with the highest log probability
        return np.argmax(log_probs, axis=1)


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
