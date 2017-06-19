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

        self._weights = []
        self._biases = []
        for i in range(len(units) - 1):
            self._weights.append((np.random.rand(units[i+1], units[i]) * 2 - 1) / np.sqrt(units[i-1]))
            self._biases.append(np.zeros(units[i+1]))

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
        self.iterations = iterations
        self.debug = debug

    @property
    def weights(self):
        return self._weights

    def evaluate(self, data, depth=None):
        # Depth can limit evaluation to a certain number of layers in the net

        if depth is None:
            depth = len(self._weights)

        for i in range(depth):

            # Dimensionality correction if processing a batch
            if np.ndim(data) == 1:
                bias = self._biases[i]
            else:
                bias = np.tile(self._biases[i][:, np.newaxis], np.shape(data)[1])

            data = self._weights[i] @ data + bias
            data = self.basis[i](data)

        return data

    def train(self, environment):

        iteration = 0
        pts = []

        # Momentum memory
        weight_update = []
        bias_update = []
        for i in range(len(self._weights)):
            weight_update.append(np.zeros(np.shape(self._weights[i])))
            bias_update.append(np.zeros(np.shape(self._biases[i])))

        converged = False
        while not converged:
            iteration += 1

            # Choose a stimulus
            [stimulus, expect] = map(np.array, environment.sample())

            # Layer derivative accumulator
            dq_dq = np.eye(np.shape(self._weights[-1])[0])

            # Train each weight set sequentially
            for layer in reversed(range(len(self._weights))):

                # ~~~~~~~ Loss derivative phase ~~~~~~~
                # stimulus = value of previous basis function or input stimulus
                s = self.evaluate(stimulus, depth=layer)
                # reinforcement = W      x s + b
                r = self._weights[layer] @ s + self._biases[layer]

                # Loss function derivative
                dln_dq = self.delta(expect, self.evaluate(stimulus), d=1) / environment.shape_input()[0]

                # Basis function derivative
                dq_dr = np.diag(self.basis[layer](r, d=1))

                # Reinforcement function derivative
                dr_dWvec = np.kron(np.identity(np.shape(self._weights[layer])[0]), s.T)

                # Chain rule for full derivative
                dln_dWvec = dln_dq @ dq_dq @ dq_dr @ dr_dWvec
                dln_db = dln_dq @ dq_dq @ dq_dr  # @ dr_db (Identity matrix)

                # Unvectorize
                dln_dW = np.reshape(dln_dWvec.T, np.shape(self._weights[layer]))

                # ~~~~~~~ Gradient descent phase ~~~~~~~
                # Same for bias and weights
                learn_rate = self.learn(iteration, self.iterations)

                # Compute bias update
                bias_gradient = -self.learn_step[layer] * dln_db
                bias_decay = self.decay_step[layer] * self.decay(self._biases[layer], d=1)
                bias_momentum = self.moment_step[layer] * bias_update[layer]

                bias_update[layer] = learn_rate * (bias_gradient + bias_decay) + bias_momentum

                # Compute weight update
                weight_gradient = -self.learn_step[layer] * dln_dW
                weight_decay = self.decay_step[layer] * self.decay(self._weights[layer], d=1)
                weight_momentum = self.moment_step[layer] * weight_update[layer]

                weight_update[layer] = learn_rate * (weight_gradient + weight_decay) + weight_momentum

                # Apply gradient descent
                self._biases[layer] += bias_update[layer]
                self._weights[layer] += weight_update[layer]

                # ~~~~~~~ Update internal state ~~~~~~~
                # Store derivative accumulation for next layer
                dr_dq = self._weights[layer]
                dq_dq = dq_dq @ dq_dr @ dr_dq

            # Exit condition
            [inputs, expectation] = map(np.array, environment.survey())
            evaluation = self.evaluate(inputs.T)[0]
            difference = np.linalg.norm(expectation.T - evaluation)
            if difference < self.epsilon:
                converged = True

            if self.iterations is not None and iteration >= self.iterations:
                break

            if self.debug:
                pts.append((iteration, difference))
                if iteration % 25 == 0:

                    # Output state of machine
                    print(str(iteration) + ': \n' + str(evaluation))

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
