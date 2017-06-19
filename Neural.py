import matplotlib.pyplot as plt
import numpy as np

import Function

plt.style.use('fivethirtyeight')


class Neural(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Delta:       sum squared, cross entropy error
    # Learn step:  learning parameter
    # Learn:       learning function
    # Decay step:  weight decay parameter
    # Decay:       weight decay function
    # Moment step: momentum strength parameter
    # Epsilon:     convergence allowance
    # Iterations:  number of iterations to run. If none, then no limit
    # Convergence: grad, newt *not implemented

    def __init__(self, units,
                 basis=Function.basis_bent, delta=Function.delta_sum_squared,
                 learn_step=1e-2, learn=Function.learn_fixed,
                 decay_step=1e-2, decay=Function.decay_NONE,
                 moment_step=1e-1,
                 epsilon=1e-2, iterations=None,
                 debug=False):

        # Weight and bias initialization. Initial random numbers are scaled by layer size.
        self.weights = []
        self.biases = []
        for i in range(len(units) - 1):
            self.weights.append((np.random.rand(units[i + 1], units[i]) * 2 - 1) / np.sqrt(units[i]))
            self.biases.append(np.zeros(units[i + 1]))

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)
        self.basis = basis

        self.delta = delta

        # Learning parameters
        if type(learn_step) is float:
            learn_step = [learn_step] * len(units)
        self.learn_step = learn_step

        self.learn = learn

        # Decay parameters
        if decay_step is None:
            decay_step = np.array(learn_step)

        if type(decay_step) is float:
            decay_step = [decay_step] * len(units)
        self.decay_step = decay_step

        self.decay = decay

        # Moment parameters
        if moment_step is None:
            moment_step = np.array(learn_step)

        if type(moment_step) is float:
            moment_step = [moment_step] * len(units)
        self.moment_step = moment_step

        self.epsilon = epsilon
        self.iteration = 0
        self.iteration_limit = iterations
        self.debug = debug

        # Internal variables to reduce time complexity of training deep nets
        self._cache_iteration = 0
        self._cache_weights = []

    def evaluate(self, data, depth=None, cache=False):
        # Depth can limit evaluation to a certain number of layers in the net

        if depth is None:
            depth = len(self.weights)

        root = 0

        # Early return if weight set already computed
        if cache is True:
            # Check for validity of cached weights
            if self.iteration == self._cache_iteration:

                # Check if cache has been computed to necessary depth
                if depth < len(self._cache_weights):
                    return self._cache_weights[depth]
                else:
                    root = len(self._cache_weights) - 1
                    data = self._cache_iteration[root]

            else:
                # Cache has been invalidated
                self._cache_weights = []

        for i in range(root, depth):

            # Dimensionality correction if processing a batch
            if np.ndim(data) == 1:
                bias = self.biases[i]
            else:
                bias = np.tile(self.biases[i][:, np.newaxis], np.shape(data)[1])

            data = self.weights[i] @ data + bias
            data = self.basis[i](data)

            if cache:
                self._cache_weights.append(data)

        return data

    def train(self, environment):

        self.iteration = 0
        pts = []

        # Momentum memory
        weight_update = []
        bias_update = []
        for i in range(len(self.weights)):
            weight_update.append(np.zeros(np.shape(self.weights[i])))
            bias_update.append(np.zeros(np.shape(self.biases[i])))

        converged = False
        while not converged:
            self.iteration += 1

            # Choose a stimulus
            [stimulus, expect] = map(np.array, environment.sample())

            # Layer derivative accumulator
            dq_dq = np.eye(np.shape(self.weights[-1])[0])

            # Train each weight set sequentially
            for layer in reversed(range(len(self.weights))):

                # ~~~~~~~ Loss derivative phase ~~~~~~~
                # stimulus = value of previous basis function or input stimulus
                s = self.evaluate(stimulus, depth=layer)
                # reinforcement = W      x s + b
                r = self.weights[layer] @ s + self.biases[layer]

                # Loss function derivative
                dln_dq = self.delta(expect, self.evaluate(stimulus), d=1) / environment.shape_input()[0]

                # Basis function derivative
                dq_dr = np.diag(self.basis[layer](r, d=1))

                # Reinforcement function derivative
                dr_dWvec = np.kron(np.identity(np.shape(self.weights[layer])[0]), s.T)

                # Chain rule for full derivative
                dln_dWvec = dln_dq @ dq_dq @ dq_dr @ dr_dWvec
                dln_db = dln_dq @ dq_dq @ dq_dr  # @ dr_db (Identity matrix)

                # Unvectorize
                dln_dW = np.reshape(dln_dWvec.T, np.shape(self.weights[layer]))

                # ~~~~~~~ Gradient descent phase ~~~~~~~
                # Same for bias and weights
                learn_rate = self.learn(self.iteration, self.iteration_limit)

                # Compute bias update
                bias_gradient = -self.learn_step[layer] * dln_db
                bias_decay = self.decay_step[layer] * self.decay(self.biases[layer], d=1)
                bias_momentum = self.moment_step[layer] * bias_update[layer]

                bias_update[layer] = learn_rate * (bias_gradient + bias_decay) + bias_momentum

                # Compute weight update
                weight_gradient = -self.learn_step[layer] * dln_dW
                weight_decay = self.decay_step[layer] * self.decay(self.weights[layer], d=1)
                weight_momentum = self.moment_step[layer] * weight_update[layer]

                weight_update[layer] = learn_rate * (weight_gradient + weight_decay) + weight_momentum

                # Apply gradient descent
                self.biases[layer] += bias_update[layer]
                self.weights[layer] += weight_update[layer]

                # ~~~~~~~ Update internal state ~~~~~~~
                # Store derivative accumulation for next layer
                dr_dq = self.weights[layer]
                dq_dq = dq_dq @ dq_dr @ dr_dq

            # Exit condition
            [inputs, expectation] = map(np.array, environment.survey())
            evaluation = self.evaluate(inputs.T)[0]
            difference = np.linalg.norm(expectation.T - evaluation)
            if difference < self.epsilon:
                converged = True

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                break

            if self.debug:
                pts.append((self.iteration, difference))
                if self.iteration % 25 == 0:

                    # Output state of machine
                    print(str(self.iteration) + ': \n' + str(evaluation))

                    plt.subplot(1, 2, 1)
                    plt.title('Error')
                    plt.plot(*zip(*pts), marker='.', color=(.9148, .604, .0945))
                    plt.pause(0.00001)
                    pts.clear()

                    plt.subplot(1, 2, 2)
                    plt.cla()
                    plt.title('Environment')
                    plt.ylim(environment.range())
                    x, y = environment.survey()
                    plt.plot(x, y, marker='.', color=(0.3559, 0.7196, 0.8637))
                    plt.plot(x, evaluation.T, marker='.', color=(.9148, .604, .0945))

                    plt.pause(0.00001)
