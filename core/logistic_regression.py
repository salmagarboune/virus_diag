"""Modèle : régression logistique."""
import numpy as np
from core.model import BaseModel, sigmoid


class LogisticRegression(BaseModel):
    def __init__(self, n_features):
        self.params = {"w": np.zeros(n_features), "b": np.array(0.0)}

    def forward(self, X):
        return sigmoid(X @ self.params["w"] + self.params["b"])

    def gradients(self, X, y):
        err = self.forward(X) - y
        return {"w": X.T @ err / len(y), "b": np.array(err.mean())}
