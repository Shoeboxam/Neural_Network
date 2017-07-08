# Learn a continuous function

# Use custom implementation:
# from Neural_Network import *

# Use Tensorflow wrapper:
from Neural_Network import *

import numpy as np
np.set_printoptions(suppress=True)


class Continuous:

    def __init__(self, funct, bounds):
        self._funct = np.vectorize(funct)
        self._bounds = bounds

        candidates = self._funct(np.linspace(*self._bounds, num=100))
        self._range = [min(candidates), max(candidates)]

    def sample(self):
        x = np.random.uniform(*self._bounds)
        return [[x], [self._funct(x)]]

    def survey(self):
        x = np.linspace(*self._bounds, num=100)
        return [np.vstack(x), self._funct(x)]

    def range(self):
        return self._range

    def size_input(self):
        return 1

    def size_output(self):
        return 1

    def plot(self, plt, predict):
        plt.ylim(self._range)
        x, y = self.survey()
        plt.plot(x, y, marker='.', color=(0.3559, 0.7196, 0.8637))
        plt.plot(x, predict.T[0], marker='.', color=(.9148, .604, .0945))

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)


environment = Continuous(lambda v: (24 * v**4 - 2 * v**2 + v), bounds=[-1, 1])

# ~~~ Create the network ~~~
init_params = {
    # Shape of network
    "units": [environment.size_input(), 15, 10, environment.size_output()],

    # Basis function(s) from Function.py
    "basis": basis_bent,

    # Weight initialization distribution
    "distribute": dist_uniform
    }

network = Neural_Network(**init_params)

# ~~~ Train the network ~~~
train_params = {
    # Source of stimuli
    "environment": environment,

    # Error function from Function.py
    "cost": cost_sum_squared,

    # Learning rate function
    "learn_step": .0001,
    "learn": learn_power,

    # Weight decay regularization function
    "decay_step": 0.0001,
    "decay": decay_NONE,

    # Momentum preservation
    "moment_step": 0.1,

    # Percent of weights to drop each training iteration
    "dropout": 0.2,

    "epsilon": .04,           # error allowance
    "iteration_limit": 500000,  # limit on number of iterations to run

    "debug": True,
    "graph": True
    }

network.train(**train_params)

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.survey()
print(network.predict(stimuli.T))
