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

    def __init__(self, units, basis=basis_logistic, distribute=dist_normal):

        self.units = units

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Construct placeholders for the input and expected output variables
            self.stimulus = tf.placeholder(tf.float32, [units[0], None], name='stimulus')
            self.expected = tf.placeholder(tf.float32, [units[-1], None], name='expected')

            # Generate the hierarchy
            self.hierarchy = self.stimulus

            self.weights = []
            self.biases = []
            for idx in range(len(units) - 1):
                self.weights.append(tf.Variable(distribute([units[idx + 1], units[idx]]), name="weight_" + str(idx)))
                self.biases.append(tf.Variable(tf.zeros((units[idx + 1])), name="bias_" + str(idx)))
                self.hierarchy = basis[idx](self.weights[-1] @ self.hierarchy + self.biases[-1][..., None])

            self.session = tf.Session(graph=self.graph)
            self.session.as_default()

    def predict(self, stimulus):
        """Stimulus evaluation"""
        return self.session.run(self.hierarchy, feed_dict={self.stimulus: stimulus})

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

        # # Learning parameters
        # if type(learn_step) is float or type(learn_step) is int:
        #     learn_step = [learn_step] * len(self.weights)
        #
        # # Decay parameters
        # if decay_step is None:
        #     decay_step = learn_step
        #
        # if type(decay_step) is float or type(decay_step) is int:
        #     decay_step = [decay_step] * len(self.weights)
        #
        # # Moment parameters
        # if moment_step is None:
        #     moment_step = learn_step
        #
        # if type(moment_step) is float or type(moment_step) is int:
        #     moment_step = [moment_step] * len(self.weights)

        # Add operations to class graph
        with self.graph.as_default():

            iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
            iteration_step_op = tf.Variable.assign_add(iteration, 1)

            train_step = convergence(learn_step).minimize(cost(self.expected, self.hierarchy))

            tf.global_variables_initializer().run(session=self.session)

        # Actual training
        pts = []
        converged = False
        while not converged:
            stimulus, expected = environment.sample(quantity=batch_size)

            self.session.run(train_step, feed_dict={self.stimulus: stimulus, self.expected: expected})

            iteration_int = self.session.run(iteration_step_op)
            if iteration_limit is not None and iteration_int >= iteration_limit:
                break

            # Stopping conditions, graphs and pretty outputs
            if (graph or epsilon or debug) and iteration_int % 50 == 0:
                [inputs, expectation] = environment.survey()
                prediction = self.predict(inputs)
                error = environment.error(expectation, prediction)

                if error < epsilon:
                    converged = True

                if debug:
                    print("Error: " + str(error))
                    # print(expectation)
                    # print(prediction)

                if graph:
                    pts.append((iteration_int, error))

                    plt.subplot(1, 2, 1)
                    plt.cla()
                    plt.title('Error')
                    plt.plot(*zip(*pts), marker='.', color=(.9148, .604, .0945))
                    plt.pause(0.00001)

                    plt.subplot(1, 2, 2)
                    plt.cla()
                    plt.title('Environment')

                    # Default graphing behaviour defined in environment.py
                    environment.plot(plt, prediction)

                    plt.pause(0.00001)

