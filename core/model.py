"""Classe de base (abstraction) de tous les modèles IA."""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Contrat commun : tout modèle expose params, forward, gradients, predict."""

    params: dict

    @abstractmethod
    def forward(self, X) -> np.ndarray:
        """Probabilités pour un batch X (n, d)."""

    @abstractmethod
    def gradients(self, X, y) -> dict:
        """Gradients de la perte pour chaque paramètre."""

    def loss(self, X, y) -> float:
        p = np.clip(self.forward(X), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict(self, x) -> float:
        """Probabilité d'infection pour UN patient."""
        return float(self.forward(np.atleast_2d(x))[0])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
