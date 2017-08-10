from .Optimize import Optimize
from ..Function import *


class ContrastiveDivergence(Optimize):

    def __init__(self, network, environment, **kwargs):

        # Default parameters
        settings = {**{
            'batch_size': 1,

            # step size
            "learn_step": 0.01,
            "learn_anneal": anneal_fixed,
            "learn_decay": 1.0,
        }, **kwargs}

        super().__init__(network, environment, **settings)

    def minimize(self):
        converged = False
        while not converged:
            self.iteration += 1

            stimulus, expectation = self.environment.sample(quantity=self.batch_size)
            # Add bias units to stimulus
            stimulus = np.vstack([stimulus, np.ones([1, self.batch_size])])

            # ____REALITY____
            # probability =              basis(W                   * s)
            probabilities = self.network.basis(self.network.weight @ stimulus)
            probabilities[-1, :] = 1  # Bias units are always one

            positive_gradient = probabilities @ stimulus.T

            # Gibbs sampling
            probabilities = probabilities > np.random.rand(*probabilities.shape)

            # ____DREAM____
            # Return to input layer by computing reconstructed stimulus
            reconstruction = self.network.basis(self.network.weight.T @ probabilities)
            reconstruction[-1, :] = 1  # Bias units are always one

            probabilities = self.network.basis(self.network.weight @ reconstruction)
            negative_gradient = probabilities @ reconstruction.T

            # ____UPDATE WEIGHTS____
            learn_rate = self.learn_anneal(self.iteration, self.learn_decay, self.iteration_limit) * self.learn_step
            self.network.weight += learn_rate * (positive_gradient - negative_gradient) / self.batch_size

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                return True

            if (self.graph or self.epsilon or self.debug) and self.iteration % self.debug_frequency == 0:
                converged = self.convergence_check()
