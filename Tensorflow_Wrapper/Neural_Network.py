import matplotlib.pyplot as plt
import os

from . import *
import tensorflow as tf

plt.style.use('fivethirtyeight')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Neural_Network(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Delta:       sum squared, cross entropy error

    def __init__(self, units, basis=basis_logistic, distribute=dist_uniform):

        self.units = units

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)

        # Construct placeholders for the input and expected output variables
        self.stimulus = tf.placeholder(tf.float32, [units[0], None], name='stimulus')
        self.expected = tf.placeholder(tf.float32, [units[-1], None], name='expected')

        # Generate the graph
        self.graph = self.stimulus

        self.weights = []
        self.biases = []
        for idx in range(len(units) - 1):
            self.weights.append(tf.Variable(distribute([units[idx + 1], units[idx]]), name="weight_" + str(idx)))
            self.biases.append(tf.Variable(tf.zeros((units[idx + 1])), name="bias_" + str(idx)))
            self.graph = basis[idx](self.weights[-1] @ self.graph + self.biases[-1][..., None])

        self.session = tf.InteractiveSession()

    def predict(self, stimulus):
        """Stimulus evaluation"""
        return self.session.run(self.graph, feed_dict={self.stimulus: stimulus})

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

    def train(self, environment, batch_size=1,
              convergence=tf.train.GradientDescentOptimizer,
              cost=cost_sum_squared,
              learn_step=1e-2, learn=learn_fixed,
              decay_step=1e-2, decay=decay_NONE,
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

        stimulus, expected = environment.sample(quantity=batch_size)

        iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
        iteration_step_op = tf.Variable.assign_add(iteration, 1)

        tf.global_variables_initializer().run()
        print(self.graph.shape)

        train_step = convergence(learn_step).minimize(cost(self.expected, self.graph))

        converged = False
        while not converged:

            if iteration == iteration_limit:
                break

            self.session.run(iteration_step_op)
            self.session.run(train_step, feed_dict={self.stimulus: stimulus, self.expected: expected})
