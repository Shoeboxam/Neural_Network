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
            self._weights.append(np.random.rand(units[i], units[i+1]) * 0.1 - 0.5)

        self._basis = basis
        self._delta = delta
        self._gamma = gamma
        self._epsilon = epsilon

    def evaluate(self, data):
        for i in range(len(self._weights)):
            data = np.matmul(data, self._weights[i])
            data = self._basis(data)
        return data

    def train(self, data, expectation):
        converged = False
        while not converged:
            for layer in range(len(self._weights)):
                # dx is used to denote the beginning of the accumulation
                # dr_dWvec =       S'                x I
                dx_dWvec = np.kron(np.transpose(data), np.identity(len(data)))

                # Accumulate derivative through all hidden layers
                for i in range(layer, len(self._weights)):
                    prediction = np.matmul(np.transpose(self._weights[i]), data) # y = W's

                    # This fails because it is taking the scalar derivative when a matrix derivative is needed.
                    # accumulator =                df_dh           * dh_dr  DIAG REMOVED             * existing
                    dx_dWvec = np.matmul(np.matmul(self._weights[i], self._basis.prime([prediction])), dx_dWvec)

                # Final error calculation
                # accumulator =       dln_df                                              * existing
                dln_dWvec = np.matmul(self._delta.prime([expectation, self.evaluate(data)]), dx_dWvec)

                # Update weights
                self._weights[layer] -= self._gamma * dln_dWvec / len(data)

            # Exit condition
            if (expectation - self.evaluate(data))**2 < self._epsilon:
                converged = True

                    # This has many bugs. I'm working on cleaning it up. Suggestions welcome.
