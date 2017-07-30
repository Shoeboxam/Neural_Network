import numpy as np
from .Array import Array


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

# PARAMETERS
tau   = 1            # Sigmoid threshold unit
alpha = 0.5          # Parametrized rectified linear unit
bank = 50            # Inverse learning function steepness


# BASIS FUNCTIONS: Regression
basis_identity  = Function('basis', 'identity',
                           [lambda x: x,
                            lambda x: np.diag(np.ones(np.shape(x)))])
basis_binary    = Function('basis', 'binary',
                           [lambda x: piecewise(x, 0, 1),
                            lambda x: diag(np.zeros(np.shape(x)))])
basis_relu      = Function('basis', 'relu',
                           [lambda x: piecewise(x, alpha * x, x),
                            lambda x: diag(piecewise(x, alpha, 1))])
basis_exponent  = Function('basis', 'exponent',
                           [lambda x: piecewise(x, alpha*(np.exp(x) - 1), x),
                            lambda x: diag(piecewise(x, alpha*np.exp(x), np.ones(np.shape(x))))])

basis_logistic  = Function('basis', 'logistic',
                           [lambda x: tau * (1 + np.exp(-x/tau))**-1,
                            lambda x: diag(np.exp(x/tau)/(np.exp(x/tau) + 1)**2)])
basis_softplus  = Function('basis', 'softplus',
                           [lambda x: np.log(1 + np.exp(x)),
                            lambda x: diag((1 + np.exp(-x))**-1)])
basis_gaussian  = Function('basis', 'gaussian',
                           [lambda x: np.exp(-x**2),
                            lambda x: diag(-2 * x * np.exp(-x**2))])

basis_tanh      = Function('basis', 'tanh',
                           [lambda x: np.tanh(x),
                            lambda x: diag(1 - np.tanh(x)**2)])
basis_arctan    = Function('basis', 'arctan',
                           [lambda x: np.arctan(x),
                            lambda x: diag(1 / (x**2 + 1))])
basis_sinusoid  = Function('basis', 'sinusoid',
                           [lambda x: np.sin(x),
                            lambda x: diag(np.cos(x))])
basis_sinc      = Function('basis', 'sinc',
                           [lambda x: piecewise_origin(x, np.sin(x) / x, 0),
                            lambda x: diag(piecewise_origin(x, np.cos(x) / x - np.sin(x) / x**2, 0))])

basis_softsign  = Function('basis', 'softsign',
                           [lambda x: x / (1 + np.abs(x)),
                            lambda x: diag(1 / (1 + np.abs(x))**2)])
basis_bent      = Function('basis', 'bent',
                           [lambda x: (np.sqrt(x**2 + 1) - 1) / 2 + x,
                            lambda x: diag(x / (2*np.sqrt(x**2 + 1)) + 1)])


# BASIS FUNCTIONS: Classification
def softmax(x):
    if x.ndim == 1:
        temp = np.exp(x - x.max())
        return temp / np.sum(temp)
    else:
        cuts = []
        for idx in range(x.shape[-1]):
            temp = np.exp(x[..., idx] - x[..., idx].max())
            cuts.append(temp / np.sum(temp, axis=0))
        return Array(np.stack(cuts, 1))

basis_softmax   = Function('basis', 'SMax',
                           [softmax,
                            lambda x: diag(softmax(x)) - softmax(x) @ softmax(x).T])


# COST FUNCTIONS
cost_sum_squared    = Function('cost', 'SSE',  # Same as RSS and SSR
                               [lambda O, P: sum((O - P)**2),
                                lambda O, P: -2 * (O - P)])

cost_cross_entropy  = Function('cost', 'CEE',
                               [lambda O, P: (O * np.log(P)) + (1 - O) * np.log(1 - P),
                                lambda O, P: (P - O)])

cost_softmax_CEE    = Function('cost', 'SMCEE',
                               [lambda O, P: (O * np.log(softmax(P))) + (1 - O) * np.log(1 - softmax(P)),
                                lambda O, P: softmax(P) - O])

# REGULARIZATION DECAY FUNCTIONS
decay_L1   = Function('decay', 'L1',
                      [lambda x: np.linalg.norm(x), lambda x: piecewise(x, -1, 1)])

decay_L2   = Function('decay', 'L2',
                      [lambda x: x.T @ x, lambda x: 2*x])

decay_L12  = Function('decay', 'L12',
                      [lambda x: decay_L1(x) + decay_L2(x), lambda x: decay_L1(x, d=1) + decay_L2(x, d=1)])

decay_NONE = Function('decay', 'NONE',
                      [lambda x: np.zeros([x.shape[1]] * 2), lambda x: np.zeros(x.shape)])


# ANNEALING FUNCTIONS (learning rate)
anneal_fixed   = Function('learn', 'fixed',
                         [lambda t, i: 1])

anneal_linear  = Function('learn', 'linear',
                         [lambda t, i: 1 - t/i])

anneal_inverse = Function('learn', 'inverse',
                         [lambda t, i: bank / (bank + t)])

anneal_power   = Function('learn', 'power',
                         [lambda t, i: np.exp(t/i)])

anneal_invroot = Function('learn', 'invroot',
                         [lambda t, i: 1 / np.sqrt(t)])


# DISTRIBUTION FUNCTIONS
dist_uniform = Function('dist', 'uniform',
                        [lambda x, y: np.random.uniform(low=-1, high=1, size=(x, y))])

dist_normal  = Function('dist', 'normal',
                        [lambda x, y: np.random.normal(loc=0, scale=1, size=(x, y))])


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


# Diagonalize first dimension of an n-dimensional array
def diag(array):
    if array.ndim == 1:
        return Array(np.diag(array))
    else:
        elements = []
        for idx in range(array.shape[-1]):
            elements.append(diag(array[..., idx]))
        return Array(np.stack(elements, array.ndim))

# Sanity check to ensure diag correctly embeds a 2D matrix along diagonal of 3D matrix
# def check_diag(A):
#     correct = True
#     A_diagon = diag(A)
#
#     for rowid in range(A.shape[1]):
#         if not np.allclose(np.diag(A.T[rowid]), A_diagon[..., rowid]):
#             correct = False
#
#     return correct
