import numpy as np

from .Optimize import Optimize
from ..Array import Array
from ..Function import *


class Backpropagation(Optimize):
    def __init__(self, network, environment, **kwargs):

        self.network = network
        self.environment = environment

        # Default parameters
        settings = {**{
            "cost": cost_sum_squared,

            # Batch size:  number of samples per training epoch
            "batch_size": 1,

            # step size
            "learn_step": 0.01,
            # modifies step size over time
            "anneal": anneal_power,

            # Weight regularizer (disabled by default)
            "regularize_step": 0.0,
            "regularizer": reg_L2,

            # Percent of weights to drop each training iteration
            "dropout_step": 0.0,
            "dropout": 0.0,
        }, **kwargs}

        super().__init__(network, environment, **settings)

        self.learn_step = self._broadcast(self.learn_step)
        self.regularize_step = self._broadcast(self.regularize_step)

        self._cached_iteration = -1
        self._cached_gradient = []
        self._cached_stimulus = []

    def minimize(self):
        iterate = np.vectorize(self.iterate)

        converged = False
        while not converged:
            self.iteration += 1

            # Update the weights
            learn_rate = self.anneal(self.iteration, self.iteration_limit) * self.learn_step
            iterate(learn_rate, list(range(len(learn_rate))))

            # Apply weight regularization
            for reg_step, weight in zip(self.regularize_step, self.network.weights):
                if reg_step:
                    weight += reg_step * self.regularizer(weight, d=1)

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                break

            if (self.graph or self.epsilon or self.debug) and self.iteration % self.debug_frequency == 0:
                [inputs, expectation] = map(Array, self.environment.survey())
                prediction = self.network.predict(inputs)
                error = self.environment.error(expectation, prediction)

                if error < self.epsilon:
                    converged = True

                if self.debug:
                    self.post_debug(error=error, prediction=prediction)

                if self.graph:
                    self.post_graph(error, prediction)

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

            # Reinforcement function derivative
            dr_dWvec_i = []
            for feature in np.vstack((s[layer], bias)).T:
                dr_dWvec_i.append(np.kron(np.identity(self.network.weights[layer].shape[0]), feature))
            dr_dWvec = Array(np.stack(dr_dWvec_i, axis=2))

            # Chain rule for full derivative
            # Matrix operators have left-to-right associativity
            dln_dWvec = dln_dq @ dq_dq @ dq_dr @ dr_dWvec

            # Unvectorize
            dln_dW = np.reshape(dln_dWvec.T, [*self.network.weights[layer].shape, self.batch_size])

            self._cached_gradient.insert(0, np.average(dln_dW, axis=2))

            # ~~~~~~~ Update internal state ~~~~~~~
            # Store derivative accumulation for next layer
            dr_dq = self.network.weights[layer][..., :-1]
            dq_dq = dq_dq @ dq_dr @ dr_dq

        self._cached_iteration = self.iteration
        return self._cached_gradient

    def _propagate(self, data):
        """Stimulus evaluation with support for dropout."""

        if self._cached_iteration == self.iteration:
            return self._cached_stimulus

        bias = np.ones([1, self.batch_size])
        self._cached_stimulus = [data]

        for idx in range(len(self.network.weights)):

            # Dropout
            if self.dropout_step is not None:
                # Drop nodes from network
                data *= np.random.binomial(1, (1.0 - self.dropout_step), size=data.shape)
                # Resize remaining nodes to compensate for loss of nodes
                data = data.astype(float) * (1.0 / (1 - self.dropout_step))

            #  r = basis          (W                 * s)
            data = self.network.basis[idx](self.network.weights[idx] @ np.vstack((data, bias)))

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

        settings = {**{'decay': 0.9, 'wedge': 0.1e8}, **settings}
        super().__init__(network, environment, **settings)

        self.decay = self._broadcast(self.decay)
        self.update = [Array(np.ones(theta.shape)) for theta in network.weights]
        self.grad_square = [Array(np.zeros([theta.shape[0]] * 2)) for theta in network.weights]
        self.update_square = [Array(np.zeros([theta.shape[0]] * 2)) for theta in network.weights]

    def iterate(self, rate, l):
        """THIS METHOD IS NOT CONVERGING"""
        # EQN 14: https://arxiv.org/pdf/1212.5701.pdf
        # Rate is derived
        self.grad_square[l] = self.decay[l] * self.grad_square[l] + \
                              (1 - self.decay[l]) * self.gradient[l] @ self.gradient[l].T

        rate = (np.diag(self.update_square[l]) + self.wedge)[..., None]
        self.update[l] = -(rate / np.sqrt(np.diag(self.grad_square[l]) + self.wedge)[..., None]) * self.gradient[l]

        # Prepare for next iteration
        self.update_square[l] = self.decay[l] * self.update_square[l] + \
                                (1 - self.decay[l]) * self.update[l] @ self.update[l].T
        self.network.weights[l] += self.update[l]


class RMSprop(Backpropagation):
    def __init__(self, network, environment, **settings):
        settings = {**{'decay': 0.9, 'wedge': 0.1e8}, **settings}
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
            'wedge': 0.1e8
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
            'wedge': 0.1e8
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
