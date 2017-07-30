import numpy as np
from .Array import Array


class Optimizer(object):
    def __init__(self, **kwargs):
        self._evaluator = getattr(self, kwargs['method'])

        if kwargs['method'] is 'momentum':
            self.update = Array(np.zeros(kwargs['shape']))
            self.decay = kwargs['decay']

        if kwargs['method'] is 'nesterov':
            self.update = Array(np.zeros(kwargs['shape']))
            self.decay = kwargs['decay']

        if kwargs['method'] is 'adagrad':
            self.epsilon = kwargs['epsilon']

        if kwargs['method'] is 'adadelta':
            self.update = Array(np.ones(kwargs['shape']))
            self.decay = kwargs['decay']
            self.epsilon = kwargs['epsilon']
            self.grad_square = Array(np.zeros([kwargs['shape'][0]] * 2))
            self.update_square = Array(np.zeros([kwargs['shape'][0]] * 2))

        if kwargs['method'] is 'rmsprop':
            self.decay = kwargs['decay']
            self.epsilon = kwargs['epsilon']
            self.grad_square = Array(np.zeros([kwargs['shape'][0]] * 2))

        if kwargs['method'] is 'adam':
            self.decay_1 = kwargs['decay_first_moment']
            self.decay_2 = kwargs['decay_second_moment']
            self.epsilon = kwargs['epsilon']
            self.grad = Array(np.ones(kwargs['shape']))
            self.grad_square = Array(np.zeros([kwargs['shape'][0]] * 2))

        if kwargs['method'] is 'adamax':
            self.decay_1 = kwargs['decay_first_moment']
            self.decay_2 = kwargs['decay_second_moment']
            self.grad = Array(np.ones(kwargs['shape']))
            self.second_moment = 0.

        if kwargs['method'] is 'nadam':
            self.decay_1 = kwargs['decay_first_moment']
            self.decay_2 = kwargs['decay_second_moment']
            self.epsilon = kwargs['epsilon']
            self.grad = Array(np.ones(kwargs['shape']))
            self.grad_square = Array(np.zeros([kwargs['shape'][0]] * 2))

    def __call__(self, rate, grad, t):
        return self._evaluator(rate, grad, t)

    @staticmethod
    def gradient_descent(rate, grad, t):
        return -rate * grad

    def momentum(self, rate, grad, t):
        self.update = -grad + self.decay * self.update
        return rate * self.update

    def nesterov(self, rate, grad, t):
        # SECTION 3.5: https://arxiv.org/pdf/1212.0901v2.pdf
        update_old = self.update
        self.update = self.decay * self.update - rate * grad
        return self.decay * (self.update - update_old) + self.update

    def adagrad(self, rate, grad, t):
        # Normalize gradient
        return -rate * grad / (np.sqrt(np.diag(grad @ grad.T)) + self.epsilon)[..., None]

    def adadelta(self, rate, grad, t):
        # EQN 14: https://arxiv.org/pdf/1212.5701.pdf
        # Rate is derived
        self.grad_square = self.decay * self.grad_square + (1 - self.decay) * grad @ grad.T

        rate = (np.diag(self.update_square) + self.epsilon)[..., None]
        self.update = -rate / np.sqrt(np.diag(self.grad_square) + self.epsilon)[..., None] * grad

        # Prepare for next iteration
        self.update_square = self.decay * self.update_square + (1 - self.decay) * self.update @ self.update.T
        return self.update

    def rmsprop(self, rate, grad, t):
        # SLIDE 29: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        self.grad_square = self.decay * self.grad_square + (1 - self.decay) * grad @ grad.T
        return -rate * grad / (np.sqrt(np.diag(self.grad_square)) + self.epsilon)[..., None]

    def adam(self, rate, grad, t):
        # Adaptive moment estimation: https://arxiv.org/pdf/1412.6980.pdf
        self.grad = self.decay_1 * self.grad + (1 - self.decay_1) * grad
        self.grad_square = self.decay_2 * self.grad_square + (1 - self.decay_2) * grad @ grad.T

        first_moment = self.grad / (1 - self.decay_1**t)
        second_moment = self.grad_square / (1 - self.decay_2**t)

        return -rate * first_moment / (np.sqrt(np.diag(second_moment) + self.epsilon)[..., None])

    def adamax(self, rate, grad, t):
        # Adaptive moment estimation: https://arxiv.org/pdf/1412.6980.pdf
        self.grad = self.decay_1 * self.grad + (1 - self.decay_1) * grad
        self.second_moment = max(self.decay_2 * self.second_moment, np.linalg.norm(grad, ord=np.inf))

        first_moment = self.grad / (1 - self.decay_1**t)
        return -rate * first_moment / self.second_moment

    def nadam(self, rate, grad, t):
        # Nesterov adaptive moment estimation: http://cs229.stanford.edu/proj2015/054_report.pdf
        self.grad = self.decay_1 * self.grad + (1 - self.decay_1) * grad
        self.grad_square = self.decay_2 * self.grad_square + (1 - self.decay_2) * grad @ grad.T

        first_moment = self.grad / (1 - self.decay_1**t)
        second_moment = self.grad_square / (1 - self.decay_2**t)

        nesterov = (self.decay_1 * first_moment + (1 - self.decay_1) * grad / (1 - self.decay_1**t))
        return -rate * nesterov / (np.sqrt(np.diag(second_moment) + self.epsilon)[..., None])


# Optimizer presets
opt_grad_descent = {
    'method': 'gradient_descent'
}

opt_momentum = {
    'method': 'momentum',
    'decay': 0.2
}

opt_nesterov = {
    'method': 'nesterov',
    'decay': 0.9
}

opt_adagrad = {
    'method': 'adagrad',
    'epsilon': .001
}

opt_adadelta = {
    'method': 'adadelta',
    'decay': 0.9,
    'epsilon': .001
}

opt_rmsprop = {
    'method': 'rmsprop',
    'decay': 0.9,
    'epsilon': 0.001
}

opt_adam = {
    'method': 'adam',
    'decay_first_moment': 0.9,
    'decay_second_moment': 0.999,
    'epsilon': 0.00000001
}

opt_adamax = {
    'method': 'adamax',
    'decay_first_moment': 0.9,
    'decay_second_moment': 0.999,
    'epsilon': 0.00000001
}

opt_nadam = {
    'method': 'adam',
    'decay_first_moment': 0.9,
    'decay_second_moment': 0.999,
    'epsilon': 0.00000001
}