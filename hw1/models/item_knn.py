import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=50, shrink=10):
        self.neighbors, self.shrink = neighbors, shrink

    def fit(self, positive):
        self.history = positive.astype(np.float64)
        gram = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(gram))
        denom = norms[:, None] * norms[None, :] + self.shrink
        sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        np.fill_diagonal(sim, 0)
        n = sim.shape[0]
        neighbors = np.argsort(-sim, axis=0, kind="stable")[:min(self.neighbors, n - 1)]
        cols = np.broadcast_to(np.arange(n), neighbors.shape)
        self.weights = sparse.csr_matrix((sim[neighbors, cols].ravel(),
                                         (neighbors.ravel(), cols.ravel())), shape=(n, n))
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()

    def recommend(self, users, seen, k=10):
        scores = np.asarray(self.score(users), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite model scores")
        rows, cols = seen[users].nonzero()
        scores[rows, cols] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError("Not enough unseen candidates")
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]
