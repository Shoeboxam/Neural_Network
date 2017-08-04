from .Optimize import Optimize
from ..Function import *

import tensorflow as tf


class Backpropagation(Optimize):
    def __init__(self, network, environment, **kwargs):

        # Default parameters
        settings = {**{
            "cost": cost_sum_squared,

            # Batch size:  number of samples per training epoch
            "batch_size": 1,

            # step size
            "learn_step": 0.01,
            # modifies step size over time
            "learn_anneal": anneal_fixed,

            # Weight regularizer (disabled by default)
            "regularize_step": 0.0,
            "regularizer": reg_L2,

            "noise_size": 0.0,
            "anneal_noise": anneal_invroot,

            # Percent of weights to drop each training iteration
            "dropout_step": 0.0,
            "dropconnect_step": 0.0,
        }, **kwargs}
        super().__init__(network, environment, **settings)

        self.learn_step = self._unbroadcast(self.learn_step)
        self.regularize_step = self._unbroadcast(self.regularize_step)

    def minimize(self):
        # --- Define Loss ---
        with self.network.graph.as_default():
            iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
            iteration_step_op = tf.Variable.assign_add(iteration, 1)

            learn_rate = tf.Variable(1., name='learn_rate', trainable=False, dtype=tf.float64)
            learn_rate_step_op = tf.Variable.assign(learn_rate, self.learn_anneal(self.iteration, self.iteration_limit))

            # Primary gradient loss
            tf.add_to_collection('losses', self.learn_step *
                                 self.cost(self.network.expected, self.network.hierarchy_train))

            # Weight decay losses
            for idx, layer in enumerate(tf.get_collection('weights')):
                regular_shape = [int(self.network.expected.shape[0]), self.batch_size]
                regular = tf.tile(self.regularizer(layer)[..., None, None], regular_shape)
                tf.add_to_collection('losses', self.regularize_step * regular)

            # Combine weight decay and gradient losses
            loss = tf.add_n(tf.get_collection('losses'), name='loss')

            # Use optimization method with given settings to minimize loss
            train_step = self.optimizer(learn_rate, **self.optimizer_args).minimize(loss)

            tf.global_variables_initializer().run(session=self.network.session)

        # --- Actual Training Portion ---
        converged = False
        while not converged:
            [stimulus, expected] = self.environment.sample(quantity=self.batch_size)

            parameters = {
                self.network.stimulus: stimulus,
                self.network.expected: expected,
                self.network.dropout: 1 - self.dropout_step
            }

            self.network.session.run(train_step, feed_dict=parameters)
            self.network.session.run(learn_rate_step_op)

            # --- Debugging and graphing ---
            # Exit condition
            iteration_int = self.network.session.run(iteration_step_op)
            if self.iteration_limit is not None and iteration_int >= self.iteration_limit:
                break

            if (self.graph or self.epsilon or self.debug) and iteration_int % self.debug_frequency == 0:
                converged = self.convergence_check()


# --------- Specific gradient update methods ---------
class GradientDescent(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'optimizer': tf.train.GradientDescentOptimizer,
            'optimizer_args': {}
        }, **settings}
        super().__init__(network, environment, **settings)


class Momentum(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'optimizer': tf.train.MomentumOptimizer,
            'optimizer_args': {settings.get('decay', default=0.2)}
        }, **settings}
        super().__init__(network, environment, **settings)

        self.optimizer_args['decay'] = self._unbroadcast(self.optimizer_args['decay'])


class Nesterov(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'optimizer': tf.train.MomentumOptimizer,
            'optimizer_args': {
                'momentum': self._unbroadcast(settings.get('decay', default=0.9)),
                'use_nesterov': True
            }
        }, **settings}
        super().__init__(network, environment, **settings)


class Adagrad(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'optimizer': tf.train.AdagradOptimizer
        }, **settings}
        super().__init__(network, environment, **settings)


class Adadelta(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'optimizer': tf.train.AdadeltaOptimizer,
            'optimizer_args': {
                'rho': self._unbroadcast(settings.get('decay', default=0.9)),
                'epsilon': self._unbroadcast(settings.get('wedge', default=0.1e-8))
            }
        }, **settings}
        super().__init__(network, environment, **settings)


class RMSprop(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'optimizer': tf.train.RMSPropOptimizer,
            'optimizer_args': {
                'decay': self._unbroadcast(settings.get('decay', default=0.9)),
                'epsilon': self._unbroadcast(settings.get('wedge', default = 0.1e-8))
            }
        }, **settings}
        super().__init__(network, environment, **settings)


class Adam(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'optimizer': tf.train.AdamOptimizer,
            'optimizer_args': {
                'beta1': self._unbroadcast(settings.get('decay_first_moment', default=0.9)),
                'beta2': self._unbroadcast(settings.get('decay_second_moment', default=0.999)),
                'epsilon': settings.get('wedge', default=1e-8)
            }
        }, **settings}
        super().__init__(network, environment, **settings)


class Adamax(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{
            'optimizer': tf.contrib.keras.optimizers.Adamax,
            'optimizer_args': {
                'beta_1': self._unbroadcast(settings.get('decay_first_moment', default=0.9)),
                'beta_2': self._unbroadcast(settings.get('decay_second_moment', default=0.999)),
                'epsilon': self._unbroadcast(settings.get('wedge', default=1e-8))
            }
        }, **settings}
        super().__init__(network, environment, **settings)


class Nadam(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{
            'optimizer': tf.contrib.keras.optimizers.Nadam,
            'optimizer_args': {
                'beta_1': self._unbroadcast(settings.get('decay_first_moment', default=0.9)),
                'beta_2': self._unbroadcast(settings.get('decay_second_moment', default=0.999)),
                'epsilon': self._unbroadcast(settings.get('wedge', default=1e-8))
            }
        }, **settings}
        super().__init__(network, environment, **settings)


# Does not exist
# class Quickprop(Backpropagation):
#     def __init__(self, network, environment, **settings):
#
#         settings = {**{'optimizer': None}, **settings}
#         super().__init__(network, environment, **settings)

# Exists via scipy wrapper, but has non-standard interface
# class L_BFGS(Backpropagation):
#     def __init__(self, network, environment, **settings):
#
#         settings = {**{'optimizer': None}, **settings}
#         super().__init__(network, environment, **settings)
