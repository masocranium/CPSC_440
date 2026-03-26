import numpy as np


class LDA:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # y should be integers from 0 to k-1
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)

        if y.ndim == 2:
            y = y.squeeze(1)
        y_counts = np.bincount(y)
        zero_classes = y_counts == 0

        n, d = X.shape
        assert y.shape == (n,)
        self.n_components_ = k = len(y_counts)

        self.coef_ = np.full((k, d), np.nan)  # w
        self.intercept_ = np.full((k,), np.nan)  # b

        # MLE estimates
        self.pi_ = pi = y_counts / n
        self.means_ = means = np.array([
            X[y == c].mean(axis=0) if y_counts[c] > 0 else np.zeros(d)
            for c in range(k)
        ])
        self.mu_ = pi @ means  # global mean

        # Pooled covariance
        Sigma = sum(
            (X[y == c] - means[c]).T @ (X[y == c] - means[c])
            for c in range(k) if y_counts[c] > 0
        ) / n + 1e-8 * np.eye(d)
        self.Sigma_ = Sigma

        # Discriminant: score_c(x) = x^T Sigma^{-1} mu_c - 0.5 mu_c^T Sigma^{-1} mu_c + log pi_c
        
        Sigma_inv_means = np.linalg.solve(Sigma, means.T).T  # (k, d)

        for c in range(k):
            if y_counts[c] == 0:
                continue
            self.coef_[c] = Sigma_inv_means[c]
            self.intercept_[c] = -0.5 * means[c] @ Sigma_inv_means[c] + np.log(pi[c])

        # make extra sure we never predict a class that had zero train points
        self.coef_[zero_classes] = 0
        self.intercept_[zero_classes] = -np.inf

        return self

    def predict(self, X):
        # [n, d] times [d, k] + [new, k]
        scores = X @ self.coef_.T + self.intercept_[np.newaxis, :]
        return np.nanargmax(scores, axis=1)

    def transform(self, X, dims=None):
        if dims is None:
            dims = self.n_components_ - 1
        if dims >= self.n_components_:
            raise ValueError(
                f"LDA only has {self.n_components_-1=} meaningful directions"
            )

        raise NotImplementedError()
