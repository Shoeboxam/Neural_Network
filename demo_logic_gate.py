# Learn a logic gate

# Use custom implementation:
from Jacobian_Chain import *

# Use Tensorflow wrapper:
# from Tensorflow_Wrapper import *

import itertools
import math
import numpy as np
np.set_printoptions(suppress=True, linewidth=10000)


class Logic_Gate:

    def __init__(self, expectation):
        bit_length = math.log(np.shape(expectation)[0], 2)
        if bit_length % 1 != 0:
            raise TypeError('Length of expectation must be a power of two.')

        self._expectation = expectation
        self._environment = np.array([i for i in itertools.product([-1, 1], repeat=int(bit_length))])

    def sample(self, quantity=1):
        choice = np.random.randint(np.shape(self._environment)[0], size=quantity)
        return self._environment[choice].T, self._expectation[choice].T

    def survey(self, quantity=None):
        # Since the domain of logic gates tends to be so small, all elements are returned in the survey
        return [self._environment.T, self._expectation.T]

    def size_input(self):
        return np.shape(self._environment)[1]

    def size_output(self):
        return np.shape(self._expectation)[1]

    def plot(self, plt, predict):
        data = np.zeros((2**self.size_input(), 2))
        predict = np.clip(predict[0], -1, 1)

        for idx, index_bits in enumerate((self._environment.astype(float) + 1) / 2):
            out = 0
            for bit in index_bits.astype(int):
                out = (out << 1) | bit
            data[out, 1] = predict[idx]
        data[:, 0] = self._expectation.T[0]

        plt.imshow(data, cmap='hot', interpolation='nearest')

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)

# environment = Logic_Gate(np.array([[-1], [1], [1], [-1]]))
environment = Logic_Gate(np.array([[-1], [1], [1], [-1], [1], [-1], [-1], [-1]]))
# environment = Logic_Gate(np.array([[1], [-1], [-1], [1], [-1], [-1], [1], [-1]]))

# Notice: The plot will only graph the first dimension of an n-dimensional input.
# environment = Logic_Gate(np.array([[-1, -1], [1, -1], [1, -1], [-1, 1]]))

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
    "optimizer": opt_adadelta,

    # Source of stimuli
    "environment": environment,
    "batch_size": 8,

    # Error function from Function.py
    "cost": cost_cross_entropy,

    # Learning rate
    "learn_step": .0001,
    "anneal": anneal_invroot,

    # Weight decay regularization function
    "decay_step": 0.00001,
    "decay": decay_NONE,

    # Percent of weights to drop each training iteration
    "dropout": 0.,

    "epsilon": .04,           # error allowance
    "iteration_limit": None,  # limit on number of iterations to run

    "debug": True,
    "graph": True
    }

network.train(**train_params)

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.survey()
print(expectation)
print(network.predict(stimuli))
