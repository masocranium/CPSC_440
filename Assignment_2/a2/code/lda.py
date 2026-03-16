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

        raise NotImplementedError()

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
