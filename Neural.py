import numpy as np
import Function


class Neural(object):
    # Units: List of quantity of nodes per layer
    # Basis: sig, rect
    # Delta: sse, logistic
    # Gamma: learning rate
    # Epsilon: convergence allowance
    # Convergence: grad, newt *not implemented

    def __init__(self, units, basis='sig', delta='sse', gamma=1e-1, epsilon=1e-4):
        self._weights = []
        for i in range(len(units) - 1):
            self._weights.append(np.random.rand(*units[i:i+1]))

        self._basis = basis
        self._delta = delta
        self._gamma = gamma
        self._epsilon = epsilon

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, name):
        if name == 'sig':
            self._basis = Function.basis_sigmoid
        else:
            self._basis = Function.basis_rectilinear

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, name):
        if name == 'sse':
            self._delta = Function.delta_linear
        else:
            self._delta = Function.delta_logistic

    @property
    def weights(self):
        return self._weights

    def evaluate(self, data):
        for i in range(len(self._weights)):
            data = np.matmul(data, self._weights[i])
            data = self._basis(data)
        return data

    def train(self, data, expectation):
        converged = False
        while not converged:
            for layer in range(len(self._weights)):
                # Where dx denotes matrix accumulation
                dx_dWvec = np.kron(numpy.ndarray.flatten(data),
                                   np.identity(np.size(np.matmul(np.transpose(self._weights[layer]), data))))

                # Accumulate derivative through all hidden layers
                for i in range(layer, len(self._weights)):
                    r = np.matmul(self._weights[i], data)
                    dx_dWvec = np.matmul(np.matmul(self._weights[i], np.diag(self._basis.prime(r))), dx_dWvec)

                # Final error calculation
                dln_dWvec = np.matmul(self._delta(expectation, self.evaluate(data)), dx_dWvec)

                # Update weights
                self._weights[layer] = self._weights[layer] - self._gamma * dln_dWvec

            if (expectation - self.evaluate(data))**2 < self._epsilon:
                converged = True

                    # This is not tested. Going to have to think about how to clean it up.
