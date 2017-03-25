import itertools
import math
import numpy as np


class Environment(object):

    def sample(self):
        # Return a single element and label from the event environment
        pass

    def survey(self):
        # Return collection of elements and labels from the event environment
        pass

    def shape_input(self):
        # Dimensions of expected input
        pass

    def shape_output(self):
        # Dimensions of expected output
        pass


class Logic_Gate(Environment):

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

    def shape_input(self):
        return np.shape(self._environment)

    def shape_output(self):
        return np.shape(self._expectation)