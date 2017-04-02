import matplotlib.pyplot as plt
import numpy as np

from Feed_Forward import Function

plt.style.use('fivethirtyeight')


class Neural(object):
    # Units: List of quantity of nodes per layer
    # Basis: sigmoidal, rectilinear
    # Delta: linear, logistic
    # Gamma: learning rate
    # Epsilon: convergence allowance
    # Convergence: grad, newt *not implemented

    def __init__(self, units,
                 basis=Function.basis_bent, delta=Function.delta_linear,
                 gamma=1e-2, epsilon=1e-2, regularization=Function.reg_NONE, debug=False):
        self._weights = []
        for i in range(len(units) - 1):
            self._weights.append((np.random.rand(units[i+1], units[i]) * 2 - 1) / np.sqrt(units[i-1]))
        self._weights[0] = np.c_[self._weights[0], np.zeros(units[1])]  # Bias units

        if type(basis) != list:
            basis = [basis] * len(units)
        self.basis = basis
        self.delta = delta

        if type(gamma) == float:
            gamma = [gamma] * len(units)
        self.gamma = gamma

        self.epsilon = epsilon
        self.debug = debug

    @property
    def weights(self):
        return self._weights

    def evaluate(self, data, depth=-1):
        # Depth can limit evaluation to a certain number of layers in the net

        # Add bias units:
        if np.ndim(data) == 1:
            data = np.r_[data, 1]
        else:
            data = np.c_[data, np.ones([np.shape(data)[0], 1])].T

        if depth == -1:
            depth = len(self._weights)

        for i in range(depth):
            data = self._weights[i] @ data

            # Do not transform on the final layer
            # if i != len(self._weights)-1:
            data = self.basis[i](data)

        return data

    def train(self, environment):

        iteration = 0       # DEBUG
        pts = []

        converged = False
        while not converged:

            # Choose a stimulus
            [stimulus, expect] = environment.sample()

            # Layer derivative accumulator
            dr_dr = np.eye(np.shape(self._weights[-1])[0])

            # Delta accumulator
            dln_dW = []
            for i in range(len(self._weights)):
                dln_dW.append(np.zeros(np.shape(self._weights[i])))

            # Train each weight set sequentially
            for layer in reversed(range(len(self._weights)-1)):
                dr_dWvec = np.kron(np.identity(np.shape(self._weights[layer])[0]),  # I
                                   self.evaluate(stimulus, depth=layer).T)          # S' at current layer

                # Notice: input to prediction is dependent on its depth within the net
                # reinforcement = W      x s
                r = self._weights[layer] @ self.evaluate(stimulus, depth=layer)
                # Interior layer accumulation
                dr_dr = dr_dr @ self._weights[layer+1] @ np.diag(self.basis[layer].prime(r))

                # Final error derivative
                dln_dr = self.delta.prime(expect, self.evaluate(stimulus)) / environment.shape_input()[0]

                dln_dWvec = dln_dr @ dr_dr @ dr_dWvec
                dln_dW[layer] += np.reshape(dln_dWvec, np.shape(self._weights[layer]))

                # Update weights
                self._weights[layer] -= self.gamma[layer] * dln_dW[layer]

            # Exit condition
            [inputs, expectation] = environment.survey()
            evaluation = self.evaluate(inputs)
            difference = np.average(np.abs(expectation - np.fliplr(evaluation.T)))
            if difference < self.epsilon:
                converged = True

            if self.debug:
                pts.append((iteration, difference))
                if iteration % 25 == 0:

                    # Output state of machine
                    print(str(iteration) + ': \n' + str(evaluation))

                    plt.subplot(1, 2, 1)
                    plt.title('Average Error')
                    plt.plot(*zip(*pts), marker='.', color=(.9148, .604, .0945))
                    plt.pause(0.00001)
                    pts.clear()

                    # Create environment plot
                    plt.subplot(1, 2, 2)
                    plt.cla()
                    plt.title('Environment')
                    plt.ylim(environment.range())
                    x, y = environment.survey()
                    plt.plot(x, evaluation.T, marker='.', color=(.9148, .604, .0945))

                    plt.plot(x, y)
                    plt.pause(0.00001)

                iteration += 1
