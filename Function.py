import numpy as np


class Function(object):
    def __init__(self, evaluator, evaluator_prime):
        self._evaluator = evaluator
        self._evaluator_prime = evaluator_prime

    def __call__(self, arguments):
        return self._evaluator(*arguments)

    def prime(self, arguments):
        return self._evaluator_prime(*arguments)

# BASIS FUNCTIONS
basis_sigmoid     = Function(lambda x: (1 + np.exp(-x))**-1, lambda x: x * (1-x))
basis_rectilinear = Function(lambda x: np.log(1 + np.exp(x)), lambda x: (1 + np.exp(-x))**-1)

# DELTA FUNCTIONS
delta_linear      = Function(lambda O, P: (O - P)**2, lambda O, P: -2 * np.transpose(O - P))
delta_logistic    = Function(lambda O, P: np.abs(O * np.log(basis_sigmoid([P])) + (1 - O) * np.log(1 - basis_sigmoid([P]))),
                             lambda O, P: np.transpose(np.zeros(np.shape(O)) + basis_sigmoid([P]) * (1 - basis_sigmoid([P]))))
