"""Prétraitement des données."""
import numpy as np


class Standardizer:
    """Centre et réduit les variables (moyenne 0, écart-type 1)."""

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd


def train_test_split(X, y, test_size=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    n_test = int(len(X) * test_size)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]
