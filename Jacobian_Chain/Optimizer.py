import numpy as np
from .Array import Array


class Optimizer(object):
    def __init__(self, **kwargs):
        self._evaluator = getattr(self, kwargs['method'])

        # Ensure history is initialized with dummy values
        if kwargs['method'] in ['momentum', 'nesterov']:
            self.update = Array(np.zeros(kwargs['shape']))
        elif kwargs['method'] is 'adadelta':
            self.update = Array(np.ones(kwargs['shape']))

        if kwargs['method'] in ['adadelta', 'rmsprop']:
            self.grad_square = Array(np.zeros([kwargs['shape'][0]] * 2))

        if kwargs['method'] is 'adadelta':
            self.update_square = Array(np.zeros([kwargs['shape'][0]] * 2))

        # Set forgetfulness parameter
        if kwargs['method'] in ['momentum', 'nesterov', 'rmsprop', 'adadelta']:
            self.moment = kwargs['moment']

        # Prevents division by zero
        if kwargs['method'] in ['adagrad', 'adadelta', 'rmsprop']:
            if 'epsilon' in kwargs.keys():
                self.epsilon = kwargs['epsilon']
            else:
                self.epsilon = .000001

    def __call__(self, rate, grad):
        return self._evaluator(rate, grad)

    @staticmethod
    def gradient_descent(rate, grad):
        return -rate * grad

    def momentum(self, rate, grad):
        self.update = -grad + self.moment * self.update
        return rate * self.update

    def nesterov(self, rate, grad):
        # SECTION 3.5: https://arxiv.org/pdf/1212.0901v2.pdf
        update_old = self.update
        self.update = self.moment * self.update - rate * grad
        return self.moment * (self.update - update_old) + self.update

    def adagrad(self, rate, grad):
        # Normalize gradient
        return -rate * grad / self._rms(grad)

    def adadelta(self, rate, grad):
        # EQN 14: https://arxiv.org/pdf/1212.5701.pdf
        # rate is derived
        self.grad_square = self.moment * self.grad_square + (1 - self.moment) * grad @ grad.T

        rate = (np.diag(self.update_square) + self.epsilon)[..., None]
        self.update = -rate / np.sqrt(np.diag(self.grad_square) + self.epsilon)[..., None] * grad

        self.update_square = self.moment * self.update_square + (1 - self.moment) * self.update @ self.update.T

        return self.update

    def rmsprop(self, rate, grad):
        # SLIDE 29: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        self.grad_square = self.moment * self.grad_square + (1 - self.moment) * grad @ grad.T
        return -rate * grad / (np.sqrt(np.diag(self.grad_square)) + self.epsilon)[..., None]

    def _rms(self, data):
        """Matrix root-mean-square"""
        return (np.sqrt(np.diag(data @ data.T)) + self.epsilon)[..., None]


# Optimizer presets
opt_grad_descent = {
    'method': 'gradient_descent'
}

opt_momentum = {
    'method': 'momentum',
    'moment': 0.2
}

opt_nesterov = {
    'method': 'nesterov',
    'moment': 0.9
}

opt_adagrad = {
    'method': 'adagrad',
    'epsilon': .001
}

opt_adadelta = {
    'method': 'adadelta',
    'moment': 0.9,
    'epsilon': .001
}

opt_rmsprop = {
    'method': 'rmsprop',
    'epsilon': 0.001,
    'moment': 0.9
}