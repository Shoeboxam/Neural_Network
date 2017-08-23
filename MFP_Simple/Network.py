# Core data structure for a multilayer feedforward perceptron network
# This implementation is restricted to 'caterpillar' function graphs
# This implementation shares the same interface as the MFP_TF network

import matplotlib.pyplot as plt
from .Function import *
import json


class MFPsimple(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...

    def __init__(self, units, basis):

        # Weight and bias initialization (bias is included inside weight matrix)
        self.weights = []
        for idx in range(len(units) - 1):
            self.weights.append(np.random.normal(loc=0, scale=1, size=(units[idx + 1], units[idx] + 1)))

        # Broadcast basis function, so that each layer has one
        if type(basis) is not list:
            basis = [basis] * len(units)

        self.basis = basis

    def predict(self, data):
        """Stimulus evaluation"""

        for idx in range(len(self.weights)):
            #  r = basis          (W                 * s)
            data = self.basis[idx](self.weights[idx] @ np.append(data, [1]))

        return data
