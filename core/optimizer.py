"""Optimiseur : met à jour les paramètres du modèle."""


class GradientDescent:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for k in params:
            params[k] = params[k] - self.lr * grads[k]
