import numpy as np


class Optimizer(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._evaluator = getattr(self, kwargs['method'])

        if kwargs['method'] is 'rmsprop':
            self._mean_square = np.zeros([kwargs['update_prev'].shape[0]] * 2)

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
        return -grad / (np.sqrt(np.diag(grad @ grad.T)) + self.epsilon)[..., None]

    def rmsprop(self, grad):
        self._mean_square = self.forget * self._mean_square + (1 - self.forget) * grad @ grad.T
        return -grad / (np.sqrt(np.diag(self._mean_square)) + self.epsilon)[..., None]
