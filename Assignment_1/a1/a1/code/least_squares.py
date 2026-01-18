import numpy as np
from utils import euclidean_dist_squared


class LeastSquares:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        self.w = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        return X @ self.w


class LeastSquaresBias:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # add bias
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        self.w = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)
        self.bias = self.w[-1]


    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        return X @ self.w[:-1] + self.bias # predict with bias is Xw + b which has dims (n,1) as expected.

def gaussianRBF_feats(X, bases, sigma):
    # Calculate squared euclidean distance between every X and every basis
    # shapes: X is (n, d), bases is (m, d) -> dists is (n, m)
    dists = euclidean_dist_squared(X, bases)
    
    # Calculate Gaussian RBF feature: exp(-||x-z||^2 / 2sigma^2)
    phi = np.exp(-dists / (2 * sigma**2))
    
    return phi # Phi has shape (n, m) where n is number of samples and m is number of bases



class LeastSquaresRBFL2:
    def __init__(self, X=None, y=None, lam=1, sigma=1):
        self.lam = lam
        self.sigma = sigma
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # Use all training points as RBF centers
        self.bases = X
        # Compute RBF features
        Phi = gaussianRBF_feats(X, self.bases, self.sigma)
        n, m = Phi.shape

        # Closed-form solution with L2 regularization
        self.w = np.linalg.solve(Phi.T @ Phi + self.lam * np.eye(m), Phi.T @ y)

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        # Compute RBF features for new data
        Phi = gaussianRBF_feats(X, self.bases, self.sigma) # n x m
        return Phi @ self.w
    

