import matplotlib.pyplot as plt

from . import *
import tensorflow as tf

plt.style.use('fivethirtyeight')


class Neural_Network(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Delta:       sum squared, cross entropy error

    def __init__(self, units, basis=basis_logistic, distribute=dist_uniform):

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)

        # Construct the network graph, starting with the input stimulus
        self.graph = tf.placeholder(tf.float32, [None, units[0]])

        for it in range(len(units) - 1):
            weights = tf.Variable(distribute([units[it], units[it + 1]]))
            bias = tf.Variable(distribute([units[it + 1]]))
            self.graph = basis[it](self.graph @ weights + bias)

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
              cost=cost_sum_squared,
              learn_step=1e-2, learn=learn_fixed,
              decay_step=1e-2, decay=decay_NONE,
              moment_step=1e-1, dropout=0,
              epsilon=1e-2, iteration_limit=None,
              debug=False, graph=False):

        session = tf.InteractiveSession()

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

        iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
        iteration_step_op = tf.Variable.assign_add(iteration, 1)

        train_step = convergence(learn_step * learn(iteration)).minimize(loss)

        converged = False
        while not converged:

            if iteration == iteration_limit:
                break
            session.run(iteration_step_op)

            batch_x
            session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
