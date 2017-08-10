# Learn a logic gate

# Use Restricted Boltzmann Machine:
from RBM import *


import itertools
import math
import numpy as np
np.set_printoptions(suppress=True, linewidth=10000)


class LogicGate:

    def __init__(self, expectation):
        bit_length = math.log(np.shape(expectation)[0], 2)
        if bit_length % 1 != 0:
            raise TypeError('Length of expectation must be a power of two.')

        self._expectation = expectation
        self._environment = np.array([i for i in itertools.product([0, 1], repeat=int(bit_length))])

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
        predict = np.clip(predict[0], 0, 1)

        for idx, index_bits in enumerate((self._environment.astype(float) + 1) / 2):
            out = 0
            for bit in index_bits.astype(int):
                out = (out << 1) | bit
            data[out, 1] = predict[idx]
        data[:, 0] = self._expectation.T[0]

        plt.imshow(data, cmap='hot', interpolation='nearest')

    @staticmethod
    def error(expect, predict):
        print(expect)
        print(predict)
        return np.linalg.norm(expect - predict)

# environment = LogicGate(np.array([[0], [1], [1], [0]]))
# environment = LogicGate(np.array([[0], [1], [1], [0], [1], [0], [0], [0]]))
environment = LogicGate(np.array([[1], [0], [0], [1], [0], [0], [1], [0]]))

# ~~~ Create the network ~~~
network_params = {
    # Shape of network
    'input_nodes': environment.size_input(),
    'output_nodes': environment.size_output(),

    "basis": basis_logistic,

    # Weight initialization distribution
    "distribute": dist_normal
    }

network = RBM(**network_params)

# ~~~ Train the network ~~~
optimizer_params = {
    "batch_size": 1,

    # Learning rate
    "learn_step": .001,
    "learn_anneal": anneal_fixed,

    "epsilon": 0.04,          # error allowance
    "iteration_limit": None,  # limit on number of iterations to run

    "debug": True,
    "graph": True
    }

ContrastiveDivergence(network, environment, **optimizer_params).minimize()

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.survey()
print(expectation)
print(network.predict(stimuli))
