import numpy as np


class Optimizer(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._evaluator = getattr(self, kwargs['method'])

    def __call__(self, grad):
        return self._evaluator(grad)

    @staticmethod
    def gradient_descent(grad):
        return -grad

    def momentum(self, grad):
        update = -grad + self.momentum_step * self.update_prev
        self.update_prev = update
        return update

    def adagrad(self, grad):
        update = -grad / (np.sqrt(np.diag(self.update_prev @ self.update_prev.T)) + self.epsilon)[..., None]
        self.update_prev = update
        return update

    def rmsprop(self, grad):
        # Incomplete
        grad_outer = self.update_prev @ self.update_prev.T
        if self._mean_square is None:
            self._mean_square = grad_outer
        else:
            self._mean_square = self.forget * self._mean_square + (1 - self.forget) * grad_outer

        return -grad / 1
