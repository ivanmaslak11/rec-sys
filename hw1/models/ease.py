import numpy as np
from scipy.linalg import solve


class EASE:
    def __init__(self, reg=100.):
        self.reg = reg

    def fit(self, positive):
        self.history = positive.astype(np.float64)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        precision = solve(gram, np.eye(gram.shape[0]), assume_a="pos", check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.)
        return self

    def score(self, users):
        return self.history[users] @ self.weights

    def recommend(self, users, seen, k=10):
        scores = np.asarray(self.score(users), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite model scores")
        rows, cols = seen[users].nonzero()
        scores[rows, cols] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError("Not enough unseen candidates")
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]
