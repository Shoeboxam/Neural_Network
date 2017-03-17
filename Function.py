import numpy as np

# Threshold unit for
tau = 1


class Function(object):
    def __init__(self, evaluator, evaluator_prime):
        self._evaluator = evaluator
        self._evaluator_prime = evaluator_prime

    def __call__(self, *args):
        return self._evaluator(*args)

    def prime(self, *args):
        return self._evaluator_prime(*args)

# BASIS FUNCTIONS
basis_sigmoid   = Function(lambda x: tau * (1 + np.exp(-x/tau))**-1, lambda x: np.exp(x)/(np.exp(x) + 1)**2)
basis_softplus  = Function(lambda x: np.log(1 + np.exp(x)), lambda x: (1 + np.exp(-x))**-1)
basis_identity  = Function(lambda x: x, lambda x: np.ones(np.shape(x)))

# DELTA FUNCTIONS
delta_linear    = Function(lambda O, P: (O - P)**2,
                           lambda O, P: -2 * np.transpose(O - P))

delta_logistic  = Function(lambda O, P: (O * np.log(basis_sigmoid(P))) + (1 - O) * np.log(1 - basis_sigmoid(P)),
                           lambda O, P: (basis_sigmoid(P) - O))

