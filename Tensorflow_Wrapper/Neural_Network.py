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
        self.basis = basis

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Construct placeholders for the input and expected output variables
            self.stimulus = tf.placeholder(tf.float32, [units[0], None], name='stimulus')
            self.expected = tf.placeholder(tf.float32, [units[-1], None], name='expected')
            self.dropout = tf.placeholder(tf.float32, name='dropout')

            # Start the hierarchy
            self.hierarchy = self.stimulus
            self.hierarchy_train = self.stimulus

            # Construct and allocate variables that define the network
            for idx in range(len(units) - 1):
                weight = tf.Variable(distribute([units[idx + 1], units[idx]]), name="weight_" + str(idx))
                self.graph.add_to_collection('weights', weight)

                bias = tf.Variable(tf.zeros((units[idx + 1])), name="bias_" + str(idx))
                self.graph.add_to_collection('biases', bias)

                self.hierarchy = basis[idx](weight @ self.hierarchy + bias[..., None])
                self.hierarchy_train = tf.nn.dropout(
                    basis[idx](weight @ self.hierarchy_train + bias[..., None]), self.dropout)

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
    # Dropout:     percent of nodes to drop from each layer
    # Epsilon:     convergence allowance
    # Iterations:  number of iterations to run. If none, then no limit
    # Debug:       make graphs and log progress to console
    # Convergence: grad, newt *not implemented

    def train(self, environment, batch_size=1,
              optimizer=opt_grad_descent,
              cost=cost_sum_squared,
              learn_step=1e-2, anneal=anneal_fixed,
              decay_step=None, decay=decay_NONE, dropout=0,
              epsilon=1e-2, iteration_limit=None,
              debug=False, graph=False):

        # --- Setup training parameters ---

        # Learning parameters - Convergence methods can be tweaked for multi-layer step sizes, but seemingly not cost
        if type(learn_step) is list:
            print("TF Wrapper: Only using first learning step size.")
            learn_step = learn_step[0]

        # Decay parameters
        if decay_step is None:
            decay_step = learn_step

        if type(decay_step) is float or type(decay_step) is int:
            decay_step = [decay_step] * len(self.units)

        # --- Define Loss ---
        with self.graph.as_default():

            iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
            iteration_step_op = tf.Variable.assign_add(iteration, 1)

            learn_rate = tf.Variable(1., name='learn_rate', trainable=False, dtype=tf.float64)
            learn_rate_step_op = tf.Variable.assign(learn_rate, anneal(iteration, iteration_limit))

            # Primary gradient loss
            tf.add_to_collection('losses', learn_step * cost(self.expected, self.hierarchy_train))

            # Weight decay losses
            for idx, layer in enumerate(tf.get_collection('weights')):
                tf.add_to_collection('losses', decay_step[idx] * tf.tile(decay(layer)[..., None, None], [1, batch_size]))

            # Combine weight decay and gradient losses
            loss = tf.add_n(tf.get_collection('losses'), name='loss')

            # Use optimization method with given settings to minimize loss
            train_step = optimizer(learn_rate, optimizer_args).minimize(loss)

            tf.global_variables_initializer().run(session=self.session)

        # --- Actual training portion ---
        pts = []
        converged = False

        while not converged:
            stimulus, expected = environment.sample(quantity=batch_size)

            parameters = {
                self.stimulus: stimulus,
                self.expected: expected,
                self.dropout: 1 - dropout
            }

            self.session.run(train_step, feed_dict=parameters)
            self.session.run(learn_rate_step_op)

            # --- Debugging and graphing ---
            # Exit condition
            iteration_int = self.session.run(iteration_step_op)
            if iteration_limit is not None and iteration_int >= iteration_limit:
                break

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

