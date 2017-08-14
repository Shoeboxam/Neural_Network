import numpy as np

from ..Function import *
from .Optimize import Optimize

# The following convergence methods are made available on 'import *'
__all__ = ['GeneticAlgorithm', 'SimulatedAnnealing']


class EvolutionaryAlgorithm(Optimize):
    def __init__(self, network, environment, **kwargs):

        # Default parameters (reproduction with no mutation)
        settings = {**{
            'batch_size': 1,
            'population_size': 20,
            'cost': cost_sum_squared,
            'distribute': dist_normal,
            'debug_frequency': 500,

            # Number of individuals to collect from population each iteration
            'samples': 3,

            # Likelihood of each kind of mutation
            'chance_replace': 0,
            'chance_noise': 0,

            # Noise mutation scaling
            'noise_scale': 1.,
            'noise_anneal': anneal_power,
            'noise_decay': 0.9999,

            # Acceptance requirements
            'temperature_size': 0.5,
            'temperature_anneal': anneal_power,  # Cooling schedule
            'temperature_decay': 0.9999,

        }, **kwargs}

        super().__init__(network, environment, **settings)

        # The first individual in the population is the current value of the weights
        self.population = [self.network.weights]
        for index in range(self.population_size - 1):
            individual = [Array(self.distribute(*w.shape) / np.sqrt(w.shape[0])) for w in self.network.weights]
            self.population.append(individual)

        [stimulus, expectation] = self.environment.survey()
        self.loss = np.average(np.abs(self.cost(expectation, self._propagate(stimulus, self.network.weights))))

    def minimize(self):

        def crossover(individuals):
            # Crossover to replace weakest individual with average of parents
            candidate = []
            for layer_slice in zip(*individuals):
                candidate.append(np.average(layer_slice[1:], axis=0))
            return candidate

        def noise(individuals):
            scale = self.noise_scale * self.noise_anneal(self.iteration, self.noise_decay, self.iteration_limit)
            candidate = []
            for layer_slice in zip(*individuals):
                candidate.append(layer_slice[0] + scale * self.distribute(*layer_slice[0].shape))
            return candidate

        def replace(individuals):
            candidate = []
            for layer_slice in zip(*individuals):
                candidate.append(self.distribute(*layer_slice[0].shape) / np.sqrt(layer_slice[0].shape[0]))
            return candidate

        mutators = [crossover, noise, replace]

        probabilities = [self.chance_noise, self.chance_replace]
        probabilities.insert(0, 1 - sum(probabilities))

        # ----- Learning Loop ----- #
        converged = False
        while not converged:
            self.iteration += 1

            # Choose stimuli and individuals from population
            [stimulus, expectation] = self.environment.sample(quantity=self.batch_size)
            individual_ind = np.random.randint(len(self.population), size=self.samples)
            individuals = [self.population[idx] for idx in individual_ind]

            # Sort by fitness
            individuals.sort(key=lambda individual:
                np.average(np.abs(self.cost(self._propagate(stimulus, individual), expectation)), axis=0))

            # Mutate or crossover to create candidate
            mutator = np.random.choice(mutators, 1, p=probabilities)[0]
            candidate = mutator(individuals)

            # Decision to keep or discard the mutated candidate
            candidate_loss = np.average(np.abs(self.cost(expectation, self._propagate(stimulus, candidate))))

            if candidate_loss < self.loss:
                probability = 1
            else:
                temperature = self.temperature_size * self.temperature_anneal(self.iteration, self.temperature_decay,
                                                                              self.iteration_limit)
                probability = np.exp(-(candidate_loss - self.loss) / temperature)

            if probability > np.random.uniform(0, 1):
                if self.debug:
                    print("Updated with probability: " + str(probability))

                self.loss = candidate_loss
                # Note: I used slice assignments to modify the weights in-place. This preserves the reference
                for layer, new in zip(individuals[0], candidate):
                    layer[:] = new

            if self.iteration_limit is not None and self.iteration >= self.iteration_limit:
                return True

            if (self.graph or self.epsilon or self.debug) and self.iteration % self.debug_frequency == 0:
                [stim, exp] = self.environment.survey()

                losses = [self.cost(self._propagate(stim, self.population[idx]), exp)
                          for idx in range(len(self.population))]
                fitness = np.average(losses, axis=1)

                self.network.weights = self.population[np.argmin(Array(fitness))]
                converged = self.convergence_check()

    def _propagate(self, data, individual):
        """Evaluate stimulus for a given individual"""

        for idx in range(len(individual)):
            bias = np.ones([1, data.shape[1]])
            #  r = basis          (W                 * s)
            data = self.network.basis[idx](individual[idx] @ np.vstack([data, bias]))
        return data


class GeneticAlgorithm(EvolutionaryAlgorithm):
    # Evolutionary algorithm with crossover, mutations and replacement
    def __init__(self, network, environment, **settings):
        settings = {**{
            'population': 30,
            'samples': 3,

            'chance_replace': 0.1,
            'chance_noise': 0.2,

            'noise_scale': 1.,
            'noise_anneal': anneal_power,
            'noise_decay': 0.9999,

            # Adjust acceptance requirements
            'temperature_size': 0.5,
            'temperature_anneal': anneal_power,  # Cooling schedule
            'temperature_decay': 0.9999,
        }, **settings}
        super().__init__(network, environment, **settings)


class SimulatedAnnealing(EvolutionaryAlgorithm):
    def __init__(self, network, environment, **settings):
        settings = {**{
            'population': 1,
            'samples': 1,

            'chance_noise': 1,

            'noise_scale': 1.,
            'noise_anneal': anneal_power,
            'noise_decay': 0.9999,

            # Adjust acceptance requirements
            'temperature_size': 0.5,
            'temperature_anneal': anneal_power,  # Cooling schedule
            'temperature_decay': 0.9999,
        }, **settings}
        super().__init__(network, environment, **settings)
