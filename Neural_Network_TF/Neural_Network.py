import matplotlib.pyplot as plt
import tensorflow as tf

from . import Function

plt.style.use('fivethirtyeight')


class Neural_Network(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Delta:       sum squared, cross entropy error

    def __init__(self, units, basis=tf.sigmoid, distribute=tf.contrib.distributions.Uniform):

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)

        # Construct the network graph
        self.graph = tf.placeholder(tf.float32, [None, units[0]])

        for i in range(len(units) - 1):
            weight = tf.Variable(distribute([units[i + 1], units[i]]))
            bias = tf.Variable(distribute([units[i + 1]]))
            self.graph = basis(weight @ self.graph + bias)

        self.session = tf.InteractiveSession()

    def predict(self, data):
        """Stimulus evaluation"""

        return data

    # Environment: class with a 'sample stimulus' method
    # Learn step:  learning parameter
    # Learn:       learning function
    # Decay step:  weight decay parameter
    # Decay:       weight decay function
    # Moment step: momentum strength parameter
    # Dropout:     percent of nodes to drop from each layer
    # Epsilon:     convergence allowance
    # Iterations:  number of iterations to run. If none, then no limit
    # Debug:       make graphs and log progress to console
    # Convergence: grad, newt *not implemented

    def train(self, environment, convergence=tf.train.GradientDescentOptimizer,
              loss=Function.delta_cross_entropy,
              learn_step=1e-2, learn=Function.learn_fixed,
              decay_step=1e-2, decay=Function.decay_NONE,
              moment_step=1e-1, dropout=0,
              epsilon=1e-2, iteration_limit=None,
              debug=False, graph=False):

        # --- Setup parameters ---

        # Learning parameters
        if type(learn_step) is float or type(learn_step) is int:
            learn_step = [learn_step] * len(self.weights)

        # Decay parameters
        if decay_step is None:
            decay_step = learn_step

        if type(decay_step) is float or type(decay_step) is int:
            decay_step = [decay_step] * len(self.weights)

        # Moment parameters
        if moment_step is None:
            moment_step = learn_step

        if type(moment_step) is float or type(moment_step) is int:
            moment_step = [moment_step] * len(self.weights)

        train_step = convergence(learn_step * learn(iteration)).minimize(loss)

        iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
        iteration_step_op = tf.Variable.assign_add(iteration, 1)

        while not converged:
            if iteration == iteration_limit:
                break

            self.session.run(iteration_step_op)
