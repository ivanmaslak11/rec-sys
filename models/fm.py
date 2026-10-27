import numpy as np
from scipy import sparse
from scipy.special import expit


class FM:
    def __init__(self, factors=16, reg=1e-5, learning_rate=.03,
                 epochs=40, negatives=2, seed=42, metadata=True):
        self.factors, self.reg, self.learning_rate = factors, reg, learning_rate
        self.epochs, self.negatives, self.seed, self.metadata = epochs, negatives, seed, metadata

    def fit(self, observed, ratings, item_metadata):
        n_users, n_items = observed.shape
        self.n_users = n_users
        self.item_features = sparse.eye(n_items, format="csr")
        if self.metadata:
            self.item_features = sparse.hstack([self.item_features, item_metadata], format="csr")
        users, items = observed.nonzero()
        labels = (ratings.data >= 4).astype(float)
        rng = np.random.default_rng(self.seed)
        neg_users = np.repeat(users[labels == 1], self.negatives)
        neg_items = rng.integers(n_items, size=len(neg_users))
        bad = np.asarray(observed[neg_users, neg_items]).ravel() != 0
        while bad.any():
            neg_items[bad] = rng.integers(n_items, size=bad.sum())
            bad = np.asarray(observed[neg_users, neg_items]).ravel() != 0
        users, items = np.r_[users, neg_users], np.r_[items, neg_items]
        labels = np.r_[labels, np.zeros(len(neg_users))]
        user_x = sparse.csr_matrix((np.ones(len(users)), (np.arange(len(users)), users)),
                                   shape=(len(users), n_users))
        x = sparse.hstack([user_x, self.item_features[items]], format="csr")
        width = x.shape[1]
        self.bias = np.array(0.)
        self.linear = np.zeros(width)
        self.embeddings = rng.normal(0., .03, (width, self.factors))
        params = [self.bias, self.linear, self.embeddings]
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        self.losses = []
        for step in range(1, self.epochs + 1):
            summed = x @ self.embeddings
            pred = self.bias + x @ self.linear + .5 * np.sum(
                summed ** 2 - x.multiply(x) @ (self.embeddings ** 2), axis=1)
            error = (expit(pred) - labels) / len(labels)

            loss = np.mean(np.logaddexp(0., pred) - labels * pred)
            loss += .5 * self.reg * (np.sum(self.linear ** 2) + np.sum(self.embeddings ** 2))

            grad_v = x.T @ (error[:, None] * (x @ self.embeddings))
            grad_v -= np.asarray(x.multiply(x).T @ error)[:, None] * self.embeddings
            grads = (
                error.sum(),
                np.asarray(x.T @ error) + self.reg * self.linear,
                grad_v + self.reg * self.embeddings,
            )
            self.losses.append(float(loss))
            for p, grad, m, v in zip(params, grads, first, second):
                m *= .9
                m += .1 * grad
                v *= .999
                v += .001 * grad * grad
                p -= self.learning_rate * (m / (1. - .9 ** step)) / (
                    np.sqrt(v / (1. - .999 ** step)) + 1e-8)
        f = self.item_features
        v = self.embeddings[n_users:]
        self.item_embedding = f @ v
        self.item_bias = f @ self.linear[n_users:] + .5 * np.sum(
            self.item_embedding ** 2 - f.multiply(f) @ (v ** 2), axis=1)
        return self

    def score(self, users):
        return (self.bias + self.linear[users, None] + self.item_bias[None, :]
                + np.dot(self.embeddings[users], self.item_embedding.T))

    def recommend(self, users, seen, k=10):
        scores = np.asarray(self.score(users), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite model scores")
        rows, cols = seen[users].nonzero()
        scores[rows, cols] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError("Not enough unseen candidates")
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]
