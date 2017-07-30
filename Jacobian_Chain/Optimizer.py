import numpy as np


class Optimizer(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._evaluator = getattr(self, kwargs['method'])

        if kwargs['method'] is 'rmsprop':
            self._mean_square = np.zeros([kwargs['update'].shape[0]] * 2)

    def __call__(self, rate, grad):
        return self._evaluator(rate, grad)

    @staticmethod
    def gradient_descent(rate, grad):
        return -rate * grad

    def momentum(self, rate, grad):
        self.update = -grad + self.momentum_step * self.update
        return rate * self.update

    def nesterov(self, rate, grad):
        # SECTION 3.5: https://arxiv.org/pdf/1212.0901v2.pdf
        update_old = self.update
        self.update = self.momentum_step * self.update - rate * grad
        return self.momentum_step * (self.update - update_old) + self.update

    def adagrad(self, rate, grad):
        return -rate * grad / (np.sqrt(np.diag(grad @ grad.T)) + self.epsilon)[..., None]

    def rmsprop(self, rate, grad):
        self._mean_square = self.forget * self._mean_square + (1 - self.forget) * grad @ grad.T
        return -rate * grad / (np.sqrt(np.diag(self._mean_square)) + self.epsilon)[..., None]


# Optimizer presets
opt_grad_descent = {
    'method': 'gradient_descent'
}

opt_momentum = {
    'method': 'momentum',
    'momentum_step': 0.2
}

opt_nesterov = {
    'method': 'nesterov',
    'momentum_step': 0.9
}

opt_adagrad = {
    'method': 'adagrad',
    'epsilon': .001
}

opt_rmsprop = {
    'method': 'rmsprop',
    'epsilon': 0.001,
    'forget': 0.9
}