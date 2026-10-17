"""Mesure les performances du modèle (accuracy, précision, rappel, F1)."""
from utils import metrics


class Evaluator:
    def __init__(self, seuil=0.5):
        self.seuil = seuil

    def evaluate(self, model, X, y):
        y_pred = (model.forward(X) >= self.seuil).astype(int)
        return {"accuracy": metrics.accuracy(y, y_pred),
                "precision": metrics.precision(y, y_pred),
                "recall": metrics.recall(y, y_pred),
                "f1": metrics.f1_score(y, y_pred)}
