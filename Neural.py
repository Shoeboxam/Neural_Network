import numpy as np
import Function


class Neural(object):
    # Units: List of quantity of nodes per layer
    # Basis: sig, rect
    # Delta: sse, logistic
    # Gamma: learning rate
    # Epsilon: convergence allowance
    # Convergence: grad, newt *not implemented

    def __init__(self, units, basis=Function.basis_sigmoid, delta=Function.delta_linear, gamma=1e-1, epsilon=1e-4):
        self._weights = []
        for i in range(len(units) - 1):
            self._weights.append(np.random.rand(units[i+1], units[i]) * 0.1 - 0.05)

        self._basis = basis
        self._delta = delta
        self._gamma = gamma
        self._epsilon = epsilon

    def evaluate(self, data, depth=-1):
        # Depth can limit evaluation to a certain number of layers in the net
        if depth == -1:
            depth = len(self._weights)

        for i in range(depth):
            data = np.matmul(self._weights[i], data)
            data = self._basis([data])
        return data

    def train(self, data, expectation):
        converged = False
        while not converged:

            # Choose a stimulus
            choice = np.random.randint(0, len(data))
            stimulus, expect = data[choice], expectation[choice]

            # Accumulate changes to each derivative
            dln_dx = [0] * len(self._weights)
            for layer in range(len(self._weights)):
                dln_dx[layer] = np.zeros(np.shape(self._weights[layer]))

            # Train each weight set sequentially
            for layer in range(len(self._weights)):

                # accumulator =       S'                 x I
                dr_dWvec = np.kron(np.transpose(stimulus), np.identity(np.shape(self._weights[layer])[0]))

                # Accumulate derivative through all hidden layers
                for i in range(layer, len(self._weights) - 1):

                    # Notice: input to prediction is dependent on its depth within the net
                    # r =         W               * s
                    r = np.matmul(self._weights[i], self.evaluate(stimulus, depth=i))
                    # dr_dr =        (dr_new)_dh        * dh_(dr_old)
                    dr_dr = np.matmul(self._weights[i+1], np.diag(self._basis.prime([r])))
                    # accumulator =     layer * existing
                    dr_dWvec = np.matmul(dr_dr, dr_dWvec)

                # accumulator =       dln_df                                              * existing
                dln_dWvec = np.matmul(self._delta.prime([expect, self.evaluate(stimulus)]), dr_dWvec)
                dln_dx[layer] += np.reshape(dln_dWvec, np.shape(self._weights[layer])) / len(data)

                # Update weights
                self._weights[layer] -= self._gamma * dln_dx[layer]

            # Exit condition
            if self._delta(expectation, self.evaluate(stimulus)) < self._epsilon:
                converged = True

                    # This has many bugs. I'm working on cleaning it up. Suggestions welcome.
