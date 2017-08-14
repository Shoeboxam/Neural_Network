# The gradient is computed in the Backpropagation class
# The child convergence method classes implement various types of weight updates

import numpy as np

from .Optimize import Optimize
from ..Array import Array
from ..Function import *

# The following convergence methods are made available on 'import *'
__all__ = ['GradientDescent', 'Momentum', 'Nesterov', 'Adagrad', 'RMSprop', 'Adam', 'Adamax', 'Nadam', 'Quickprop']


class Backpropagation(Optimize):
    def __init__(self, network, environment, **kwargs):

        # Default parameters
        settings = {**{
            "cost": cost_sum_squared,

            # Batch size:  number of samples per training epoch
            "batch_size": 1,

            # step size
            "learn_step": 0.01,
            "learn_anneal": anneal_fixed,
            "learn_decay": 1.0,

            # Batch norm regularization decay rate (disabled by default)
            "batch_normalize": 0.0,

            # Weight decay (disabled by default)
            "regularize_step": 0.0,
            "regularizer": reg_L2,

            # Percent of weights to drop each training iteration (disabled by default)
            "dropout_step": 0.0,
            "dropconnect_step": 0.0,

            # Maximum weight size (disabled by default)
            "weight_clip": clip_soft,
            "weight_threshold": 0.0,

            # Maximum gradient size (disabled by default)
            "gradient_clip": clip_soft,
            "gradient_threshold": 0.0,

            # Gradient noise (disabled by default)
            "noise_variance": 0.0,
            "noise_anneal": anneal_inverse,
            "noise_decay": 0.5,
        }, **kwargs}

        super().__init__(network, environment, **settings)

        self.learn_step = self._broadcast(self.learn_step)
        self.learn_decay = self._broadcast(self.learn_decay)
        self.regularize_step = self._broadcast(self.regularize_step)

        self.dropout_step = self._broadcast(self.dropout_step)
        self.dropconnect_step = self._broadcast(self.dropconnect_step)

        self.weight_threshold = self._broadcast(self.weight_threshold)
        self.gradient_threshold = self._broadcast(self.gradient_threshold)

        self.noise_variance = self._broadcast(self.noise_variance)
        self.noise_decay = self._broadcast(self.noise_decay)

        self._cached_iteration = -1
        self._cached_gradient = []
        self._cached_stimulus = []

        # Cache the gradient down to batch norm separately
        self._cached_gradient_batch = []

    def minimize(self):

        iterate = np.vectorize(self.iterate)
        learn_anneal = np.vectorize(self.learn_anneal)

        converged = self.convergence_check()
        while not converged:
            self.iteration += 1

            # Update the weights
            learn_rate = learn_anneal(self.iteration, self.learn_decay, self.iteration_limit) * self.learn_step
            iterate(learn_rate, list(range(len(learn_rate))))

            for l, weight in enumerate(self.network.weights):
                # Weight decay regularizer
                if self.regularize_step[l]:
                    self.network.weights[l] -= self.regularize_step[l] * self.regularizer(self.network.weights[l])

                # Max norm constraint - 5.1 http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
                if self.weight_threshold[l]:
                    self.network.weights[l] = self.weight_clip(weight, self.weight_threshold[l])

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                return true

            if (self.graph or self.epsilon or self.debug) and self.iteration % self.debug_frequency == 0:
                converged = self.convergence_check()

        return converged

    @property
    def gradient(self):

        if self._cached_iteration == self.iteration:
            return self._cached_gradient
        self._cached_gradient.clear()

        bias = np.ones([1, self.batch_size])

        # Choose a stimulus
        stimulus, expect = map(Array, self.environment.sample(quantity=self.batch_size))

        # stimulus = value of previous basis function or input stimulus, with bias units
        s = self._propagate(stimulus)

        # Layer derivative accumulator
        dq_dq = Array(np.eye(self.network.weights[-1].shape[0]))

        # Loss function derivative
        dln_dq = self.cost(expect, s[-1], d=1)[None, ...]

        # Train each weight set sequentially
        for layer in reversed(range(len(self.network.weights))):

            # reinforcement = W             * s
            r = self.network.weights[layer] @ np.vstack((s[layer], bias))

            # Basis function derivative
            dq_dr = Array(self.network.basis[layer](r, d=1))

            # Final reinforcement derivative
            dr_dW = np.vstack((s[layer], bias))[None]

            # Chain rule for full derivative
            # Matrix operators have left-to-right associativity
            # This is a tensor product simplification to avoid the use of the kron product
            dln_dW = (dln_dq @ dq_dq @ dq_dr).T @ dr_dW

            self._cached_gradient.insert(0, np.average(dln_dW, axis=2))

            # ~~~~~~~ Update internal state ~~~~~~~
            # Store derivative accumulation for next layer
            dr_dq = self.network.weights[layer][..., :-1]
            dq_dq = dq_dq @ dq_dr @ dr_dq

        for l in range(len(self._cached_gradient)):

            # https://arxiv.org/pdf/1511.06807.pdf
            if self.noise_variance[l]:
                # Schedule decay in variance
                variance = self.noise_variance[l] * self.noise_anneal(self.iteration, self.noise_decay, self.iteration_limit)
                self._cached_gradient[l] += np.random.normal(0, variance, [*self.network.weights[l].shape])

            # https://arxiv.org/pdf/1211.5063.pdf
            if self.gradient_threshold[l]:
                self._cached_gradient[l] = self.gradient_clip(self._cached_gradient[l], self.gradient_threshold[l])

        self._cached_iteration = self.iteration
        return self._cached_gradient

    def _propagate(self, data):
        """Stimulus evaluation with support for dropout."""

        if self._cached_iteration == self.iteration:
            return self._cached_stimulus

        bias = np.ones([1, self.batch_size])

        # Normalize and scale/shift initial input (batch norm regularization)
        if self.batch_normalize:
            data = (data - self.network.mean[0]) / np.sqrt(self.network.variance[0] + 1e-8)
            data = self.network.scale[0] * data + self.network.shift[0]
        self._cached_stimulus = [data]

        for l in range(len(self.network.weights)):

            # Dropout - https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
            if self.dropout_step[l]:
                # Drop nodes from network
                data[np.random.binomial(1, self.dropout_step, size=data.shape[0]).astype(bool)] = 0
                # Resize remaining nodes to compensate for loss of nodes
                data = data.astype(float) / (1 - self.dropout_step)

            # Dropconnect - https://cs.nyu.edu/~wanli/dropc/dropc.pdf
            if self.dropconnect_step[l]:
                # Drop connections from the network
                mask = np.random.binomial(1, (1.0 - self.dropconnect_step), size=self.network.weights[l].shape).astype(float)
                # Resize remaining nodes to compensate for loss of nodes
                mask *= mask / (1.0 - self.dropconnect_step)
                data = self.network.weights[l] * mask @ np.vstack((data, bias))

            else:
                #  r = W                       * s
                data = self.network.weights[l] @ np.vstack((data, bias))

            # Batch norm regularizer update  - https://arxiv.org/pdf/1502.03167.pdf
            if self.batch_normalize:
                mean = np.average(self._cached_stimulus[l])
                self.network.mean[l] = self.batch_normalize * self.network.mean[l] + (1 - self.batch_normalize) * mean

                variance = np.var(self._cached_stimulus[l])
                self.network.variance[l] = self.batch_normalize * self.network.variance[l] + (1 - self.batch_normalize) * variance

                # Normalize output of linear transform, then scale and shift
                data = (data - self.network.mean[l + 1]) / np.sqrt(self.network.variance[l + 1] + 1e-8)
                data = self.network.scale[l + 1] * data + self.network.shift[l + 1]

            # Important: apply activation function to transform into next subspace
            data = self.network.basis[l](data)
            self._cached_stimulus.append(data)

        return self._cached_stimulus

    def iterate(self, i, rate):
        raise NotImplementedError("Optimizer is an abstract base class")


# --------- Specific gradient update methods ---------
class GradientDescent(Backpropagation):
    def iterate(self, rate, l):
        self.network.weights[l] -= rate * self.gradient[l]


class Momentum(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{'decay': 0.2}, **settings}
        super().__init__(network, environment, **settings)

        self.decay = self._broadcast(self.decay)
        self.update = [Array(np.zeros(theta.shape)) for theta in network.weights]

    def iterate(self, rate, l):
        self.update[l] = self.gradient[l] + self.decay[l] * self.update[l]
        self.network.weights[l] -= rate * self.update[l]


class Nesterov(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{'decay': 0.9}, **settings}
        super().__init__(network, environment, **settings)

        self.decay = self._broadcast(self.decay)
        self.update = [Array(np.zeros(theta.shape)) for theta in network.weights]

    def iterate(self, rate, l):
        # SECTION 3.5: https://arxiv.org/pdf/1212.0901v2.pdf
        update_old = self.update[l].copy()
        self.update[l] = self.decay[l] * self.update[l] - rate * self.gradient[l]
        self.network.weights[l] += self.decay[l] * (self.update[l] - update_old) + self.update[l]


class Adagrad(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{'wedge': 0.001}, **settings}
        super().__init__(network, environment, **settings)

    def iterate(self, rate, l):
        # Normalize gradient
        norm = (np.sqrt(np.diag(self.gradient[l] @ self.gradient[l].T)) + self.wedge)[..., None]
        self.network.weights[l] -= rate * self.gradient[l] / norm


class Adadelta(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{'decay': 0.9, 'wedge': 1e-8}, **settings}
        super().__init__(network, environment, **settings)

        self.decay = self._broadcast(self.decay)
        self.update = [Array(np.ones(theta.shape)) for theta in network.weights]
        self.grad_square = [Array(np.zeros([theta.shape[0]] * 2)) for theta in network.weights]
        self.update_square = [Array(np.zeros([theta.shape[0]] * 2)) for theta in network.weights]

    def iterate(self, rate, l):
        # TODO: This method is not converging
        # EQN 14: https://arxiv.org/pdf/1212.5701.pdf
        # Rate is derived
        self.grad_square[l] = self.decay[l] * self.grad_square[l] + \
                              (1 - self.decay[l]) * self.gradient[l] @ self.gradient[l].T

        rate = np.sqrt(np.diag(self.update_square[l]) + self.wedge)[..., None]
        self.update[l] = -(rate / np.sqrt(np.diag(self.grad_square[l]) + self.wedge)[..., None]) * self.gradient[l]
        print((rate / np.sqrt(np.diag(self.grad_square[l]) + self.wedge)[..., None]).shape)
        self.network.weights[l] -= self.update[l]

        # Prepare for next iteration
        self.update_square[l] = self.decay[l] * self.update_square[l] + \
                                (1 - self.decay[l]) * self.update[l] @ self.update[l].T


class RMSprop(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{'decay': 0.9, 'wedge': 1e-8}, **settings}
        super().__init__(network, environment, **settings)

        self.decay = self._broadcast(self.decay)
        self.grad_square = [Array(np.zeros([theta.shape[0]] * 2)) for theta in network.weights]

    def iterate(self, rate, l):
        # SLIDE 29: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        self.grad_square[l] = self.decay[l] * self.grad_square[l] + \
                              (1 - self.decay[l]) * self.gradient[l] @ self.gradient[l].T

        self.network.weights[l] -= rate * self.gradient[l] / \
                                   (np.sqrt(np.diag(self.grad_square[l])) + self.wedge)[..., None]


class Adam(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{
            'decay_first_moment': 0.9,
            'decay_second_moment': 0.999,
            'wedge': 1e-8
        }, **settings}
        super().__init__(network, environment, **settings)

        self.decay_first_moment = self._broadcast(self.decay_first_moment)
        self.decay_second_moment = self._broadcast(self.decay_second_moment)
        self.grad_cache = [Array(np.ones(theta.shape)) for theta in network.weights]
        self.grad_square = [Array(np.zeros([theta.shape[0]] * 2)) for theta in network.weights]

    def iterate(self, rate, l):
        # Adaptive moment estimation: https://arxiv.org/pdf/1412.6980.pdf
        self.grad_cache[l] = self.decay_first_moment[l] * self.grad_cache[l] + \
                             (1 - self.decay_first_moment[l]) * self.gradient[l]

        self.grad_square[l] = self.decay_second_moment[l] * self.grad_square[l] + \
                              (1 - self.decay_second_moment[l]) * self.gradient[l] @ self.gradient[l].T

        first_moment = self.grad_cache[l] / (1 - self.decay_first_moment[l] ** self.iteration)
        second_moment = self.grad_square[l] / (1 - self.decay_second_moment[l] ** self.iteration)

        self.network.weights[l] -= rate * first_moment / \
                                   (np.sqrt(np.diag(second_moment) + self.wedge)[..., None])


class Adamax(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{
            'decay_first_moment': 0.9,
            'decay_second_moment': 0.999
        }, **settings}
        super().__init__(network, environment, **settings)

        self.decay_first_moment = self._broadcast(self.decay_first_moment)
        self.decay_second_moment = self._broadcast(self.decay_second_moment)
        self.grad_cache = [Array(np.ones(theta.shape)) for theta in network.weights]
        self.second_moment = [0.0] * len(network.weights)

    def iterate(self, rate, l):
        # Adaptive moment estimation: https://arxiv.org/pdf/1412.6980.pdf
        self.grad_cache[l] = self.decay_first_moment[l] * self.grad_cache[l] + \
                             (1 - self.decay_first_moment[l]) * self.gradient[l]
        self.second_moment[l] = max(self.decay_second_moment[l] * self.second_moment[l],
                                    np.linalg.norm(self.gradient[l], ord=np.inf))

        first_moment = self.grad_cache[l] / (1 - self.decay_first_moment[l] ** self.iteration)
        self.network.weights[l] -= rate * first_moment / self.second_moment[l]


class Nadam(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{
            'decay_first_moment': 0.9,
            'decay_second_moment': 0.999,
            'wedge': 1e-8
        }, **settings}
        super().__init__(network, environment, **settings)

        self.decay_first_moment = self._broadcast(self.decay_first_moment)
        self.decay_second_moment = self._broadcast(self.decay_second_moment)
        self.grad_cache = [Array(np.ones(theta.shape)) for theta in network.weights]
        self.grad_square = [Array(np.zeros([theta.shape[0]] * 2)) for theta in network.weights]

    def iterate(self, rate, l):
        # Nesterov adaptive moment estimation: http://cs229.stanford.edu/proj2015/054_report.pdf
        self.grad_cache[l] = self.decay_first_moment[l] * self.grad_cache[l] + \
                             (1 - self.decay_first_moment[l]) * self.gradient[l]
        self.grad_square[l] = self.decay_second_moment[l] * self.grad_square[l] + \
                              (1 - self.decay_second_moment[l]) * self.gradient[l] @ self.gradient[l].T

        first_moment = self.grad_cache[l] / (1 - self.decay_first_moment[l] ** self.iteration)
        second_moment = self.grad_square[l] / (1 - self.decay_second_moment[l] ** self.iteration)

        nesterov = (self.decay_first_moment[l] * first_moment +
                    (1 - self.decay_first_moment[l]) * self.gradient[l] /
                    (1 - self.decay_first_moment[l] ** self.iteration))
        self.network.weights[l] -= rate * nesterov / \
                                   (np.sqrt(np.diag(second_moment) + self.wedge)[..., None])


class Quickprop(Backpropagation):
    def __init__(self, network, environment, **settings):

        settings = {**{'maximum_growth_factor': 1.2}, **settings}
        super().__init__(network, environment, **settings)

        self.maximum_growth_factor = self._broadcast(self.maximum_growth_factor)

        self.update = [Array(np.ones(theta.shape)) for theta in network.weights]
        self.gradient_cache = [Array(np.zeros(theta.shape)) for theta in network.weights]

    def iterate(self, rate, l):
        # https://arxiv.org/pdf/1606.04333.pdf
        limit = np.abs(self.update[l]) * self.maximum_growth_factor[l]
        self.update[l] = np.clip(self.gradient[l] / (self.gradient_cache[l] - self.gradient[l]), -limit, limit)

        self.network.weights[l] -= rate * self.update[l]
        self.gradient_cache[l] = self.gradient[l].copy()


class L_BFGS(Backpropagation):
    def __init__(self, network, environment, **settings):
        super().__init__(network, environment, **settings)

        self.update = [Array(np.zeros(theta.shape)) for theta in network.weights]
        self.grad_cache = [Array(np.zeros(theta.shape)) for theta in network.weights]

        self.hessian_inv = [Array(np.eye(theta.shape[0])) for theta in network.weights]

    def iterate(self, rate, l):
        """THIS METHOD IS NOT FULLY IMPLEMENTED"""
        update_delta = self.update[l] - update
        grad_delta = self.gradient[l] - self.grad_cache[l]

        alpha = (update_delta.T @ self.gradient[l]) / (grad_delta.T @ update_delta)

        self.update[l] = self.gradient[l] + self.decay[l] * self.update[l]
        self.network.weights[l] -= rate * self.update[l]