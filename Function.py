import numpy as np


class Function(object):
    def __init__(self, usage, name, evaluators):
        self.usage = usage
        self.name = name
        self._evaluator = evaluators

    def __call__(self, *args, d=0):
        return self._evaluator[d](*args)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<' + self.usage + ' ' + self.name + '>'

# PARAMETERS
tau   = 1            # Sigmoid threshold unit
alpha = 0            # Parametrized rectified linear unit
bank = 50            # Inverse learning function steepness

# BASIS FUNCTIONS: Regression
basis_identity  = Function('basis', 'identity',
                           [lambda x: x, lambda x: np.ones(np.shape(x))])
basis_binary    = Function('basis', 'binary',
                           [lambda x: piecewise(x, 0, 1), lambda x: np.zeros(np.shape(x))])
basis_relu      = Function('basis', 'relu',
                           [lambda x: piecewise(x, alpha * x, x), lambda x: piecewise(x, alpha, 1)])
basis_exponent  = Function('basis', 'exponent',
                           [lambda x: piecewise(x, alpha*(np.exp(x) - 1), x),
                            lambda x: piecewise(x, alpha*np.exp(x), np.ones(np.shape(x)))])

basis_logistic  = Function('basis', 'logistic',
                           [lambda x: tau * (1 + np.exp(-x/tau))**-1, lambda x: np.exp(-x/tau)/(np.exp(-x/tau) + 1)**2])
basis_softplus  = Function('basis', 'softplus',
                           [lambda x: np.log(1 + np.exp(x)), lambda x: (1 + np.exp(-x))**-1])
basis_gaussian  = Function('basis', 'gaussian',
                           [lambda x: np.exp(-x**2), lambda x: -2 * x * np.exp(-x**2)])

basis_tanh      = Function('basis', 'tanh',
                           [lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2])
basis_arctan    = Function('basis', 'arctan',
                           [lambda x: np.arctan(x), lambda x: 1 / (x**2 + 1)])
basis_sinusoid  = Function('basis', 'sinusoid',
                           [lambda x: np.sin(x), lambda x: np.cos(x)])
basis_sinc      = Function('basis', 'sinc',
                           [lambda x: piecewise_origin(x, np.sin(x) / x, 0),
                            lambda x: piecewise_origin(x, np.cos(x) / x - np.sin(x) / x**2, 0)])

basis_softsign  = Function('basis', 'softsign',
                           [lambda x: x / (1 + np.abs(x)), lambda x: 1 / (1 + np.abs(x))**2])
basis_bent      = Function('basis', 'bent',
                           [lambda x: (np.sqrt(x**2 + 1) - 1) / 2 + x, lambda x: x / (2*np.sqrt(x**2 + 1)) + 1])


# BASIS FUNCTIONS: Classification
basis_softmax   = Function('basis', 'SMax',
                           [lambda O, P: (np.exp(P) / np.sum(np.exp(P)))])


# DELTA FUNCTIONS
delta_sum_squared    = Function('delta', 'SSE',
                           [lambda O, P: (O - P)**2,
                            lambda O, P: -2 * np.transpose(O - P)])

delta_cross_entropy  = Function('delta', 'CEE',
                           [lambda O, P: (O * np.log(basis_sigmoid(P))) + (1 - O) * np.log(1 - basis_sigmoid(P)),
                            lambda O, P: (basis_sigmoid(P) - O)])

# REGULARIZATION FUNCTIONS
reg_L1   = Function('reg', 'L1',
                    [lambda x: np.linalg.norm(x), lambda x: piecewise(x, -1, 1)])

reg_L2   = Function('reg', 'L2',
                    [lambda x: x**2, lambda x: 2*x])

reg_NONE = Function('reg', 'NONE',
                    [lambda x: 0, lambda x: 0])


# LEARNING RATE FUNCTIONS
learn_fixed   = Function('learn', 'fixed',
                         [lambda t, i: 1])

learn_linear  = Function('learn', 'linear',
                         [lambda t, i: 1 - t/i])

learn_inverse = Function('learn', 'inverse',
                         [lambda t, i: bank / (bank + t)])

learn_power   = Function('learn', 'power',
                         [lambda t, i: np.exp(t/i)])


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
