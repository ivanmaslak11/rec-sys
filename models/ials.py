import numpy as np
from scipy.linalg import solve


class IALS:
    def __init__(self, factors=32, reg=10., alpha=20., iterations=15, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def _step(self, support, target, fixed):
        result = np.zeros((support.shape[0], self.factors))
        common = np.dot(fixed.T, fixed) + self.reg * np.eye(self.factors)
        for row in range(support.shape[0]):
            start, end = support.indptr[row:row + 2]
            ids = support.indices[start:end]
            if len(ids):
                y = fixed[ids]
                lhs = common + np.dot(self.alpha * y.T, y)
                rhs = (1. + self.alpha) * np.dot(y.T, target.data[start:end])
                result[row] = solve(lhs, rhs, assume_a="pos", check_finite=False)
        return result

    def fit(self, observed, targets):
        observed = observed.astype(np.float64).tocsr(copy=True)
        targets = targets.astype(np.float64).tocsr(copy=True)
        observed.sort_indices()
        targets.sort_indices()
        if not (np.array_equal(observed.indptr, targets.indptr) and
                np.array_equal(observed.indices, targets.indices)):
            raise ValueError("Targets and observed support must align, including zeros")
        transposed = observed.T.tocsr()
        target_t = targets.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, .05, (observed.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._step(observed, targets, self.item_factors)
            self.item_factors = self._step(transposed, target_t, self.user_factors)
        return self

    def score(self, users):
        return np.dot(self.user_factors[users], self.item_factors.T)

    def recommend(self, users, seen, k=10):
        scores = np.asarray(self.score(users), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite model scores")
        rows, cols = seen[users].nonzero()
        scores[rows, cols] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError("Not enough unseen candidates")
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]
