import numpy as np

from ..Function import *
from .Optimize import Optimize


class SimulatedAnnealing(Optimize):
    def __init__(self, network, environment, **kwargs):

        # Default parameters
        settings = {**{
            'temperature_size': 0.4,
            'temperature_anneal': anneal_invroot,  # Cooling schedule
            'cost': cost_sum_squared,
            'distribute': dist_uniform,
            "debug_frequency": 2000,
            'batch_size': 1
        }, **kwargs}

        super().__init__(network, environment, **settings)

        [stimulus, expectation] = self.environment.sample(quantity=self.batch_size)
        self.fitness = np.average(np.abs(self.cost(expectation, self._propagate(stimulus, self.network.weights))))

    def minimize(self):

        converged = False
        while not converged:
            self.iteration += 1
            temperature = self.temperature_size * self.temperature_anneal(self.iteration, self.iteration_limit)

            # Choose from population
            [stimulus, expectation] = self.environment.sample(quantity=self.batch_size)
            neighbor = Array([weight + np.random.normal(0, temperature, size=weight.shape)
                              for weight in self.network.weights])

            neighbor_fitness = np.average(np.abs(self.cost(expectation, self._propagate(stimulus, neighbor))))

            if neighbor_fitness < self.fitness:
                probability = 1
            else:
                probability = np.exp(-(neighbor_fitness - self.fitness) / temperature)

            if probability < np.random.uniform(0, 1):
                self.fitness = neighbor_fitness
                self.network.weights = neighbor

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                return True

            if (self.graph or self.epsilon or self.debug) and self.iteration % self.debug_frequency == 0:
                converged = self.convergence_check()

    def _propagate(self, data, individual):
        """Evaluate stimulus for a given individual"""

        for idx in range(len(individual)):
            bias = np.ones([1, data.shape[1]])
            #  r = basis          (W                 * s)
            data = self.network.basis[idx](individual[idx] @ np.vstack([data, bias]))
        return data
