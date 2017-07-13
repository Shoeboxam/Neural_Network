import numpy as np
import Tensorflow_Wrapper as tf


class Function(object):
    def __init__(self, usage, name, evaluator):
        self.usage = usage
        self.name = name
        self._evaluator = evaluator

    def __call__(self, *args):
        return self._evaluator(*args)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<' + self.usage + ' ' + self.name + '>'

# PARAMETERS
tau   = 1            # Sigmoid threshold unit
alpha = 0.5          # Parametrized rectified linear unit
bank = 50            # Inverse learning function steepness

# BASIS FUNCTIONS: Regression
basis_identity  = Function('basis', 'identity', tf.identity)
basis_binary    = Function('basis', 'identity', lambda x: tf.where(tf.greater(x, 0), 1, 0))

basis_relu      = Function('basis', 'relu', tf.nn.relu)
basis_exponent  = Function('basis', 'exponent', lambda x: tf.where(tf.greater(x, 0), alpha*(np.exp(x) - 1), x))

basis_logistic  = Function('basis', 'logistic', tf.nn.sigmoid)
basis_softplus  = Function('basis', 'softplus', tf.nn.softplus)
basis_gaussian  = Function('basis', 'gaussian', lambda x: tf.exp(-x**2))

basis_tanh      = Function('basis', 'tanh', tf.nn.tanh)
basis_arctan    = Function('basis', 'arctan', tf.atan)
basis_sinusoid  = Function('basis', 'sinusoid', tf.sin)
basis_sinc      = Function('basis', 'sinc', lambda x: tf.sin(x) / x)

basis_softsign  = Function('basis', 'softsign', tf.nn.softsign)
basis_bent      = Function('basis', 'bent', lambda x: (tf.sqrt(x**2 + 1) - 1) / 2 + x)


# BASIS FUNCTIONS: Classification
basis_softmax   = Function('basis', 'SMax', tf.nn.softmax)


# COST FUNCTIONS
cost_sum_squared    = Function('cost', 'SSE', tf.squared_difference)

cost_cross_entropy  = Function('cost', 'CEE',
                               lambda O, P: tf.reduce_mean(-tf.reduce_sum(O * tf.log(P), reduction_indices=[1])))
# TODO: Catch and correct the final basis layer to identity when training a classifier, change CEE to softmax CEE
cost_softmax_CE     = Function('cost', 'SMCEE',
                               lambda O, P: tf.nn.softmax_cross_entropy_with_logits(labels=O, logits=P))


# REGULARIZATION DECAY FUNCTIONS
decay_L1   = Function('decay', 'L1', lambda x: tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1), x))
decay_L2   = Function('decay', 'L2', tf.nn.l2_loss)

decay_L12  = Function('decay', 'L12', lambda x: tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1), x) + tf.nn.l2_loss(x))
decay_NONE = Function('decay', 'NONE', lambda x: 0)


# LEARNING RATE FUNCTIONS
learn_fixed   = Function('learn', 'fixed', lambda t, i: 1)

learn_linear  = Function('learn', 'linear', lambda t, i: 1 - t/i)

learn_inverse = Function('learn', 'inverse', lambda t, i: bank / (bank + t))

learn_power   = Function('learn', 'power', lambda t, i: np.exp(t/i))

learn_invroot = Function('learn', 'invroot', lambda t, i: 1 / np.sqrt(t))


# DISTRIBUTION FUNCTIONS
dist_uniform = Function('dist', 'uniform',
                        lambda shape: tf.contrib.distributions.Uniform(low=-1., high=1.).sample(shape))
dist_normal  = Function('dist', 'uniform',
                        lambda shape: tf.contrib.distributions.Normal(loc=0., scale=1.).sample(shape))
