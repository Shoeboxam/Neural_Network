import numpy as np
import Function
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


class Neural(object):
    # Units: List of quantity of nodes per layer
    # Basis: sigmoidal, rectilinear
    # Delta: linear, logistic
    # Gamma: learning rate
    # Epsilon: convergence allowance
    # Convergence: grad, newt *not implemented

    def __init__(self, units,
                 basis=Function.basis_sigmoid, delta=Function.delta_linear,
                 gamma=1e-2, epsilon=1e-2, debug=False):
        self._weights = []
        for i in range(len(units) - 1):
            self._weights.append((np.random.rand(units[i+1], units[i]) - .5) * 25/np.sqrt(units[i]))
        self._weights[0] = np.c_[self._weights[0], np.zeros(units[1])]

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
        if np.ndim(data) == 1:
            data = np.r_[data, 1]
        else:
            data = np.r_[data, np.ones([1, np.shape(data)[np.ndim(data)-1]])]

        if depth == -1:
            depth = len(self._weights)

        for i in range(depth):
            data = self._weights[i] @ data

            # Do not transform on the final layer
            if i != len(self._weights)-1:
                data = self.basis(data)

        return data

    def train(self, data, expectation):

        iteration = 0       # DEBUG
        pts = []

        converged = False
        while not converged:

            # Choose a stimulus
            choice = np.random.randint(len(data))
            [stimulus, expect] = data[choice], expectation[choice]

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
                dr_dr = dr_dr @ self._weights[layer+1] @ np.diag(self.basis.prime(r))

                # Final error derivative
                dln_dr = self.delta.prime(expect, self.evaluate(stimulus)) / len(data)

                dln_dWvec = dln_dr @ dr_dr @ dr_dWvec
                dln_dW[layer] += np.reshape(dln_dWvec, np.shape(self._weights[layer]))

                # Update weights
                self._weights[layer] -= self.gamma[layer] * dln_dW[layer]

            # Exit condition
            difference = np.sum(np.abs(expectation - self.evaluate(data.T)))
            if difference < self.epsilon:
                converged = True

            if self.debug:
                pts.append((iteration, difference))
                if iteration % 25 == 0:
                    print(str(iteration) + ': ' + str(self.evaluate(data.T)[0]))
                    plt.plot(*zip(*pts), marker='.', color=(.9148, .604, .0945))
                    plt.pause(0.00001)
                    pts.clear()

                iteration += 1
