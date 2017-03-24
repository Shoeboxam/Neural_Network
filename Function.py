import numpy as np


class Function(object):
    def __init__(self, evaluator, evaluator_prime):
        self._evaluator = evaluator
        self._evaluator_prime = evaluator_prime

    def __call__(self, *args):
        return self._evaluator(*args)

    def prime(self, *args):
        return self._evaluator_prime(*args)

# PARAMETERS
tau   = 1      # Sigmoid threshold unit
alpha = 0         # Parametrized rectified linear unit

# BASIS FUNCTIONS
basis_identity  = Function(lambda x: x, lambda x: np.ones(np.shape(x)))
basis_binary    = Function(lambda x: piecewise(x, 0, 1), lambda x: np.zeros(np.shape(x)))
basis_relu      = Function(lambda x: piecewise(x, alpha * x, x), lambda x: piecewise(x, alpha, 1))
basis_exponent  = Function(lambda x: piecewise(x, alpha*(np.exp(x) - 1), x),
                           lambda x: piecewise(x, alpha*np.exp(x), np.ones(np.shape(x))))

basis_sigmoid   = Function(lambda x: tau * (1 + np.exp(-x/tau))**-1, lambda x: np.exp(x/tau)/(np.exp(x/tau) + 1)**2)
basis_softplus  = Function(lambda x: np.log(1 + np.exp(x)), lambda x: (1 + np.exp(-x))**-1)
basis_gaussian  = Function(lambda x: np.exp(-x**2), lambda x: -2 * x * np.exp(-x**2))

basis_tanh      = Function(lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2)
basis_arctan    = Function(lambda x: np.arctan(x), lambda x: 1 / (x**2 + 1))
basis_sinusoid  = Function(lambda x: np.sin(x), lambda x: np.cos(x))
basis_sinc      = Function(lambda x: piecewise_origin(x, np.sin(x) / x, 1),
                           lambda x: piecewise_origin(x, np.cos(x) / x - np.sin(x) / x**2, 0))

basis_softsign  = Function(lambda x: x / (1 + np.abs(x)), lambda x: 1 / (1 + np.abs(x))**2)
basis_bent      = Function(lambda x: (np.sqrt(x**2 + 1) - 1) / 2 + x, lambda x: x / (2*np.sqrt(x**2 + 1)) + 1)


# DELTA FUNCTIONS
delta_linear    = Function(lambda O, P: (O - P)**2,
                           lambda O, P: -2 * np.transpose(O - P))

delta_logistic  = Function(lambda O, P: (O * np.log(basis_sigmoid(P))) + (1 - O) * np.log(1 - basis_sigmoid(P)),
                           lambda O, P: (basis_sigmoid(P) - O))


def piecewise(x, lower, upper, thresh=0):

    low_indices = np.where(x < thresh)
    if type(lower) == float or type(lower) == int:
        x[low_indices] = lower
    else:
        x[low_indices] = lower[low_indices]

    up_indices = np.where(x > thresh)
    if type(upper) == float or type(upper) == int:
        x[up_indices] = upper
    else:
        x[up_indices] = upper[up_indices]
    return x


def piecewise_origin(x, outer, inner, origin=0):
    x[np.where(x == origin)] = inner

    out_indices = np.where(x != origin)
    if type(outer) == float or type(outer) == int:
        x[out_indices] = outer
    else:
        x[out_indices] = outer[out_indices]
    return x