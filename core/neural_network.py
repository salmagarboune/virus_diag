"""Modèle : petit réseau de neurones (1 couche cachée)."""
import numpy as np
from core.model import BaseModel, sigmoid


class NeuralNetwork(BaseModel):
    def __init__(self, n_features, hidden=8, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, 0.5, (n_features, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 0.5, hidden),
            "b2": np.array(0.0),
        }

    def _hidden(self, X):
        return np.tanh(X @ self.params["W1"] + self.params["b1"])

    def forward(self, X):
        return sigmoid(self._hidden(X) @ self.params["W2"] + self.params["b2"])

    def gradients(self, X, y):
        n = len(y)
        a1 = self._hidden(X)
        err = sigmoid(a1 @ self.params["W2"] + self.params["b2"]) - y
        dz1 = err[:, None] * self.params["W2"][None, :] * (1 - a1 ** 2)
        return {"W1": X.T @ dz1 / n, "b1": dz1.mean(axis=0),
                "W2": a1.T @ err / n, "b2": np.array(err.mean())}
