# Functions specific to restricted boltzmann machines
# Adapted from MFP/Functions.py

import numpy as np


class Function(object):
    def __init__(self, usage, name, evaluators):
        self.usage = usage
        self.name = name
        self._evaluator = evaluators

    def __call__(self, *args, d=0):
        # The optional d parameter is being used to denote power of derivative
        return self._evaluator[d](*args)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<' + self.usage + ' ' + self.name + '>'


# BASIS FUNCTIONS: Regression
# Diagonalize first dimension of an n-dimensional array

tau = 1            # Sigmoid threshold unit
basis_logistic = Function('basis', 'logistic',  # Commonly known as 'Sigmoid'
                          [lambda x: tau * (1 + np.exp(-x/tau))**-1,                       # S
                           lambda x: np.diag(np.exp(x / tau) / (np.exp(x / tau) + 1) ** 2)])  # S * (1 - S)


# BASIS FUNCTIONS: Classification
def softmax(x):
    temp = np.exp(x - x.max())
    return temp / np.sum(temp)

basis_softmax = Function('basis', 'SMax',
                         [softmax,
                          lambda x: diag(softmax(x)) - softmax(x) @ softmax(x).T])


# ANNEALING FUNCTIONS (learning rate)
anneal_fixed   = Function('learn', 'fixed',
                         [lambda t, d, lim: 1])

anneal_linear  = Function('learn', 'linear',
                         [lambda t, d, lim: 1 - t/lim])

anneal_inverse = Function('learn', 'inverse',
                         [lambda t, d, lim: 1 / (d * t)])

anneal_power   = Function('learn', 'power',
                         [lambda t, d, lim: d**t])

anneal_exp     = Function('learn', 'exp',
                          [lambda t, d, lim: np.exp(-t / l)])


# DISTRIBUTION FUNCTIONS
dist_uniform = Function('dist', 'uniform',
                        [lambda *args: np.random.uniform(low=-1, high=1, size=[*args])])

dist_normal  = Function('dist', 'normal',
                        [lambda *args: np.random.normal(loc=0, scale=1, size=[*args])])
