# Learn a continuous function
from inspect import signature

# Use custom implementation:
# from Jacobian_Chain import *

from Gradient_Propagation import *

# Use Tensorflow_Wrapper wrapper:
# from Tensorflow_Wrapper import *

import numpy as np
np.set_printoptions(suppress=True)


class Continuous:

    def __init__(self, funct, bounds):
        self._size_input = len(signature(funct[0]).parameters)
        self._size_output = len(funct)

        self._funct = funct
        self._bounds = bounds

    def sample(self, quantity=1):
        # Generate random values for each input stimulus
        stimulus = []
        for idx in range(self._size_input):
            stimulus.append(np.random.uniform(low=self._bounds[idx][0], high=self._bounds[idx][1], size=quantity))

        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    def survey(self, quantity=100):
        # Generate random values for each input stimulus
        stimulus = []
        for idx in range(self._size_input):
            stimulus.append(np.linspace(start=self._bounds[idx][0], stop=self._bounds[idx][1], num=quantity))

        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    def size_input(self):
        return self._size_input

    def size_output(self):
        return self._size_output

    def plot(self, plt, predict):
        plt.ylim(self._range)
        x, y = self.survey()
        plt.plot(x, y, marker='.', color=(0.3559, 0.7196, 0.8637))
        plt.plot(x, predict.T[0], marker='.', color=(.9148, .604, .0945))

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)


environment = Continuous([lambda a, b: (24 * a**4 - 2 * b**2 + a),
                          lambda a, b: (-5 * a**3 + 2 * b**2 + b),
                          lambda a, b: (12 * a**2 + 8 * b**3 + b)], bounds=[[-1, 1]] * 2)

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
    "batch_size": 30,

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
