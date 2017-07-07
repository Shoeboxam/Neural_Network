# Learn a logic gate
from Neural_Network import *

import itertools
import math
import numpy as np
np.set_printoptions(suppress=True)


class Logic_Gate:

    def __init__(self, expectation):
        bit_length = math.log(np.shape(expectation)[0], 2)
        if bit_length % 1 != 0:
            raise TypeError('Length of expectation must be a power of two.')

        self._expectation = expectation
        self._environment = np.array([i for i in itertools.product([0, 1], repeat=int(bit_length))])

    def sample(self):
        choice = np.random.randint(np.shape(self._environment)[0])
        return self._environment[choice], self._expectation[choice]

    def survey(self):
        return [self._environment, self._expectation]

    def size_input(self):
        return np.shape(self._environment)[1]

    def size_output(self):
        return np.shape(self._expectation)[1]

    def plot(self, plt, predict):
        data = np.zeros((2**self.size_input(), 2))
        predict = np.clip(predict.T[0], 0, 1)

        for idx, index_bits in enumerate(self._environment):
            out = 0
            for bit in index_bits:
                out = (out << 1) | bit
            data[out, 1] = predict[idx]
        data[:, 0] = self._expectation.T[0]

        plt.imshow(data, cmap='hot', interpolation='nearest')

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)

# environment = Logic_Gate(np.array([[0], [1], [1], [0]]))
environment = Logic_Gate(np.array([[0], [1], [1], [0], [1], [0], [0], [0]]))
# environment = Logic_Gate(np.array([[1], [0], [0], [1], [0], [0], [1], [0]]))

# Notice: The plot will only graph the first dimension of an n-dimensional input.
# environment = Logic_Gate(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))

# ~~~ Create the network ~~~
init_params = {
    # Shape of network
    "units": [environment.size_input(), 20, 10, environment.size_output()],

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
    "learn_step": .005,
    "learn": learn_fixed,

    # Weight decay regularization function
    "decay_step": 0.0001,
    "decay": decay_L2,

    # Momentum preservation
    "moment_step": 0,

    # Percent of weights to drop each training iteration
    "dropout": 0,

    "epsilon": .04,           # error allowance
    "iteration_limit": 500000,  # limit on number of iterations to run

    "debug": True,
    "graph": True
    }

network.train(**train_params)

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.survey()
print(network.predict(stimuli.T))
