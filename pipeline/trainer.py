"""Entraîne le modèle (boucle d'apprentissage)."""


class Trainer:
    def __init__(self, optimizer, epochs=1000):
        self.optimizer = optimizer
        self.epochs = epochs
        self.history = []

    def train(self, model, X, y):
        for _ in range(self.epochs):
            grads = model.gradients(X, y)
            self.optimizer.step(model.params, grads)
            self.history.append(model.loss(X, y))
        return model
