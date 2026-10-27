import numpy as np
from .ials import IALS


class CustomALS(IALS):
    def __init__(self, half_life_days=730., **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def fit_ratings(self, observed, ratings, timestamps, cutoff):
        if not (np.array_equal(ratings.indptr, timestamps.indptr) and
                np.array_equal(ratings.indices, timestamps.indices)):
            raise ValueError("Ratings and timestamps must align")
        age_days = (cutoff - timestamps.data) / 86400.
        if np.any(age_days < 0):
            raise ValueError("Future observations in training")
        target = ratings.copy().astype(np.float64)
        target.data = ((ratings.data - 3.) / 2.) * np.exp2(-age_days / self.half_life_days)
        return super().fit(observed, target)
