import tensorflow as tf


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


# OPTIMIZATION FUNCTIONS
opt_grad_descent = Function('conv', 'grad_descent',
                            lambda r, args: tf.train.GradientDescentOptimizer(r, **args))

opt_momentum     = Function('conv', 'momentum',
                            lambda r, args: tf.train.MomentumOptimizer(r, **args))

opt_adagrad      = Function('conv', 'adagrad',
                            lambda r, args: tf.train.AdagradOptimizer(r, **args))


# BASIS FUNCTIONS: Regression
basis_identity  = Function('basis', 'identity', tf.identity)
basis_binary    = Function('basis', 'identity', lambda x: tf.where(tf.greater(x, 0), 1, 0))

basis_relu      = Function('basis', 'relu', tf.nn.relu)
basis_exponent  = Function('basis', 'exponent', lambda x: tf.where(tf.greater(x, 0), alpha*(tf.exp(x) - 1), x))

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
cost_sum_squared    = Function('cost', 'SSE', tf.losses.mean_squared_error)

cost_cross_entropy  = Function('cost', 'CEE',
                               lambda O, P: tf.reduce_mean(-tf.reduce_sum(O * tf.log(P + .01), reduction_indices=[1])))
# TODO: Catch and correct the final basis layer to identity when training a classifier, change CEE to softmax CEE
cost_softmax_CE     = Function('cost', 'SMCEE',
                               lambda O, P: tf.nn.softmax_cross_entropy_with_logits(labels=O, logits=P))


# REGULARIZATION DECAY FUNCTIONS # Returns a scalar tensor
reg_L1   = Function('decay', 'L1', lambda x: tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1.), [x]))
reg_L2   = Function('decay', 'L2', tf.nn.l2_loss)

reg_L12  = Function('decay', 'L12', lambda x: tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(1.), [x]) + tf.nn.l2_loss(x))


# ANNEALING FUNCTIONS (learning rate)
anneal_fixed    = Function('learn', 'fixed', lambda r, t, d, lim: r)

anneal_linear   = Function('learn', 'linear', lambda r, t, d, lim: 1 - t / lim)

anneal_inverse  = Function('learn', 'inverse',
                           lambda r, t, d, lim: tf.train.inverse_time_decay(r, t, 1, d, staircase=False))

anneal_power    = Function('learn', 'power',
                           lambda r, t, d, lim: tf.train.exponential_decay(r, t, 1, d, staircase=False))

anneal_exponent = Function('learn', 'exp',
                           lambda r, t, d, lim: tf.train.natural_exp_decay(r, t, 1, d, staircase=False))

anneal_poly     = Function('learn', 'poly',
                           lambda r, t, d, lim: tf.train.polynomial_decay(r, t, 1, 1e-8, d))


# DISTRIBUTION FUNCTIONS
dist_uniform = Function('dist', 'uniform',
                        lambda shape: tf.contrib.distributions.Uniform(low=-1., high=1.).sample(shape))
dist_normal  = Function('dist', 'uniform',
                        lambda shape: tf.contrib.distributions.Normal(loc=0., scale=1.).sample(shape))
