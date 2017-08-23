# Implementation of backpropagation, derivative, and gradient descent

import numpy as np

from .Optimize import Optimize
from ..Function import *


class GradientDescent(Optimize):
    def __init__(self, network, environment, **kwargs):

        # Default parameters
        settings = {**{
            "cost": cost_sum_squared,

            # Step size
            "learn_step": 0.01,
        }, **kwargs}

        super().__init__(network, environment, **settings)

        self.learn_step = self._broadcast(self.learn_step)

        self._cached_iteration = -1
        self._cached_gradient = []
        self._cached_stimulus = []

    def minimize(self):

        converged = False
        while not converged:
            self.iteration += 1

            # Update each weight set in a loop, this is the part that actually does gradient descent
            for l in range(len(self.network.weights)):
                self.network.weights[l] -= self.learn_step[l] * self.gradient[l]

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                converged = False

            if (self.graph or self.epsilon or self.debug) and self.iteration % self.debug_frequency == 0:
                converged = self.convergence_check()

        return converged

    @property
    def gradient(self):

        if self._cached_iteration == self.iteration:
            return self._cached_gradient
        self._cached_gradient.clear()

        # Choose a stimulus
        stimulus, expect = self.environment.sample(quantity=1)

        # In this simplified network, there is no batch processing, so the training data is always a vector
        stimulus = stimulus[:, 0]
        expect = expect[:, 0]

        # stimulus = value of previous basis function or input stimulus, with bias units
        s = self._propagate(stimulus)

        # Layer derivative accumulator
        dq_dq = np.eye(self.network.weights[-1].shape[0])

        # Loss function derivative
        dln_dq = self.cost(expect, s[-1], d=1).T

        # Train each weight set sequentially
        for l in reversed(range(len(self.network.weights))):

            # reinforcement = W         * s
            r = self.network.weights[l] @ np.append(s[l], [1])

            # Basis function derivative
            dq_dr = self.network.basis[l](r, d=1)

            # Final reinforcement derivative
            dr_dWvec = np.kron(np.append(s[l], [1]), np.identity(self.network.weights[l].shape[0]))
            # Chain rule to compute gradient down to weight matrix
            # Matrix operators have left-to-right associativity
            dln_dWvec = dln_dq @ dq_dq @ dq_dr @ dr_dWvec
            dln_dW = np.reshape(dln_dWvec, self.network.weights[l].shape)

            self._cached_gradient.insert(0, dln_dW)

            # ~~~~~~~ Update internal state ~~~~~~~
            # Store derivative accumulation for next layer
            dr_dq = self.network.weights[l][..., :-1]
            dq_dq = dq_dq @ dq_dr @ dr_dq

        self._cached_iteration = self.iteration
        return self._cached_gradient

    def _propagate(self, data):
        """Stimulus evaluation with internal features cached"""

        if self._cached_iteration == self.iteration:
            return self._cached_stimulus

        self._cached_stimulus = [data]

        for l in range(len(self.network.weights)):
            #  r = basis                (W                       * s)
            data = self.network.basis[l](self.network.weights[l] @ np.append(data, [1]))
            self._cached_stimulus.append(data)

        return self._cached_stimulus
