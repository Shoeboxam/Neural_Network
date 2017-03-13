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
                 gamma=1e-1, epsilon=1e-4, debug=False):
        self._weights = []
        for i in range(len(units) - 1):
            self._weights.append(np.random.rand(units[i+1], units[i]) * 0.1 - 0.05)

        self.basis = basis
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.debug = debug

    @property
    def weights(self):
        return self._weights

    def evaluate(self, data, depth=-1):
        # Depth can limit evaluation to a certain number of layers in the net
        if depth == -1:
            depth = len(self._weights)

        for i in range(depth):
            data = np.matmul(self._weights[i], data)
            data = self.basis([data])
        return data

    def train(self, data, expectation):
        converged = False
        iteration = 0       # DEBUG
        pts = []

        while not converged:

            # Choose a stimulus
            choice = np.random.randint(0, len(data))
            stimulus, expect = data[choice], expectation[choice]

            # Layer derivative accumulator
            df_dr = np.eye(np.shape(self._weights[-1])[0])

            # Train each weight set sequentially
            for layer in reversed(range(len(self._weights) - 1)):
                dr_dWvec = np.kron(np.transpose(self.evaluate(stimulus, depth=layer)),  # S' at current layer
                                   np.identity(np.shape(self._weights[layer])[0]))      # I

                # Notice: input to prediction is dependent on its depth within the net
                # reinforcement = W               * s
                r = np.matmul(self._weights[layer], self.evaluate(stimulus, depth=layer))
                # dr_dr (next layer) = W (next layer)       * one-to-one nonlinear transformation
                dr_dr = np.matmul(self._weights[layer+1], np.diag(self.basis.prime([r])))

                # Accumulate interior derivative
                df_dr = np.matmul(df_dr, dr_dr)
                # accumulator =     layers * input
                dr_dWvec = np.matmul(df_dr, dr_dWvec)

                # accumulator =       dln_df                                             * existing
                dln_dWvec = np.matmul(self.delta.prime([expect, self.evaluate(stimulus)]), dr_dWvec)
                dln_dW = np.reshape(dln_dWvec, np.shape(self._weights[layer])) / len(data)

                # Update weights
                self._weights[layer] -= self.gamma * dln_dW

            # Exit condition
            difference = np.linalg.norm(self.delta([expectation, self.evaluate(np.transpose(data))]))
            if difference < self.epsilon:
                converged = True

            if self.debug:
                print(str(iteration) + ': ' + str(self.evaluate(np.transpose(data))))
                pts.append((iteration, difference))
                if iteration % 25 == 0:
                    x, y = zip(*pts)
                    plt.plot(x, y, marker='.', color=(.9148, .604, .0945))
                    plt.pause(0.00001)
                    pts.clear()
                iteration += 1
