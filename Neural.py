import matplotlib.pyplot as plt
import numpy as np

import Function

plt.style.use('fivethirtyeight')


class Neural(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       sigmoidal, rectilinear...
    # Delta:       sum squared, cross entropy error
    # Gamma:       learning parameter
    # Learn:       learning function
    # Decay:       weight decay
    # Epsilon:     convergence allowance
    # Iterations:  number of iterations to run. If none, then no limit
    # Convergence: grad, newt *not implemented

    def __init__(self, units,
                 basis=Function.basis_bent, delta=Function.delta_sum_squared,
                 gamma=1e-2, gamma_bias=None, epsilon=1e-2, regul=Function.reg_NONE,
                 learn=Function.learn_fixed, iterations=None, decay=1.0, debug=False):

        self._weights = []
        self._biases = []
        for i in range(len(units) - 1):
            self._weights.append((np.random.rand(units[i+1], units[i]) * 2 - 1) / np.sqrt(units[i-1]))
            self._biases.append(np.zeros(units[i+1]))

        if type(basis) is not list:
            basis = [basis] * len(units)
        self.basis = basis
        self.delta = delta

        if type(gamma) is float:
            gamma = [gamma] * len(units)
        self.gamma = gamma

        if gamma_bias is None:
            gamma_bias = np.array(gamma)

        if type(gamma_bias) is float:
            gamma_bias = [gamma_bias] * len(units)
        self.gamma_bias = gamma_bias

        self.learn = learn

        self.regul = regul
        self.decay = decay

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

            # Disable transform on the final layer if computing the final training layer
            # if i != len(self._weights)-1:
            data = self.basis[i](data)

        return data

    def train(self, environment):

        iteration = 0
        pts = []

        converged = False
        while not converged:

            # Choose a stimulus
            [stimulus, expect] = map(np.array, environment.sample())

            # Layer derivative accumulator
            dq_dq = np.eye(np.shape(self._weights[-1])[0])

            # Delta accumulator
            dln_dW = []
            for i in range(len(self._weights)):
                dln_dW.append(np.zeros(np.shape(self._weights[i])))

            # Train each weight set sequentially
            for layer in reversed(range(len(self._weights))):

                # stimulus = value of previous basis function or input stimulus
                s = self.evaluate(stimulus, depth=layer)
                # reinforcement = W      x s + b
                r = self._weights[layer] @ s + self._biases[layer]

                # Loss function derivative
                dln_dq = self.delta(expect, self.evaluate(stimulus), d=1) / environment.shape_input()[0]

                # Basis function derivative
                dq_dr = dq_dq @ np.diag(self.basis[layer](r, d=1))

                # Reinforcement function derivative
                dr_dWvec = np.kron(np.identity(np.shape(self._weights[layer])[0]), s.T)  # S at current layer

                # Chain rule for full derivative
                dln_dWvec = dln_dq @ dq_dr @ dr_dWvec
                dln_db = dln_dq @ dq_dr  # * dr_db (Identity matrix)

                # Store derivative accumulation last
                dr_dq = self._weights[layer]
                dq_dq = dq_dr @ dr_dq

                # Unvectorize
                dln_dW[layer] += np.reshape(dln_dWvec, np.shape(self._weights[layer]))

                # Add regularization
                regularization = .01 * self.regul(self._weights[layer], d=1)

                # Update biases
                learn_rate = self.learn(iteration, self.iterations) * self.gamma_bias[layer]
                self._biases[layer] = self._biases[layer] * self.decay - np.array(learn_rate * dln_db)

                # Update weights
                learn_rate = self.learn(iteration, self.iterations) * self.gamma[layer]
                self._weights[layer] = self._weights[layer] * self.decay - np.array(learn_rate * dln_dW[layer]) + regularization

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

                iteration += 1
