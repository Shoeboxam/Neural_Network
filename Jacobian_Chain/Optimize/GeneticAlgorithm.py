import numpy as np

from ..Function import *
from .Optimize import Optimize


class GeneticAlgorithm(Optimize):
    def __init__(self, network, environment, **kwargs):

        self.network = network
        self.environment = environment

        # Default parameters
        settings = {**{
            'population_size': 100,
            'cost': cost_sum_squared,
            'distribute': dist_uniform,
            "debug_frequency": 2000,
            'mutation_rate': 0.5,
            'mutation_anneal': anneal_fixed,
            'batch_size': 1
        }, **kwargs}

        super().__init__(network, environment, **settings)

    def minimize(self):
        population = []
        for index in range(self.population_size):
            individual = [Array(self.distribute(*w.shape) / np.sqrt(w.shape[0])) for w in self.network.weights]
            population.append(individual)

        converged = False
        while not converged:
            self.iteration += 1

            # Choose from population
            [stimulus, expectation] = self.environment.sample(quantity=self.batch_size)
            selection = np.random.randint(len(population), size=3)

            # Compute the fitness of each individual; lower is better
            losses = Array([self.cost(self._propagate(stimulus, population[idx]), expectation) for idx in selection])
            fitness = np.average(losses, axis=1)

            # Determine which individual must be replaced
            weakest = selection[np.argmax(fitness)]
            parents = selection[np.argpartition(fitness, 2)[:2]]

            # Determine chance of mutation
            mutation_chance = self.mutation_rate * self.mutation_anneal(self.iteration, self.iteration_limit)
            mutate = np.random.binomial(1, mutation_chance)

            if mutate:
                population[weakest] = [Array(self.distribute(*w.shape) / np.sqrt(w.shape[0])) for w in self.network.weights]
            else:
                # Crossover to replace weakest individual
                child = np.average(Array([population[parents[0]], population[parents[1]]]), axis=0)
                population[weakest] = child

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                return True

            if (self.graph or self.epsilon or self.debug) and self.iteration % self.debug_frequency == 0:
                losses = [self.cost(self._propagate(stimulus, population[idx]), expectation)
                          for idx in range(len(population))]
                fitness = np.average(losses, axis=1)

                self.network.weights = population[np.argmin(Array(fitness))]
                converged = self.convergence_check()

    def _propagate(self, data, individual):
        """Evaluate stimulus for a given individual"""

        for idx in range(len(individual)):
            bias = np.ones([1, data.shape[1]])
            #  r = basis          (W                 * s)
            data = self.network.basis[idx](individual[idx] @ np.vstack([data, bias]))
        return data
