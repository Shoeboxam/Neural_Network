# Core data structure for a multilayer feedforward perceptron network
# This implementation is restricted to 'caterpillar' function graphs
# This implementation shares the same interface as the MFP_TF network

import matplotlib.pyplot as plt
from .Array import Array
from .Function import *
import json


plt.style.use('fivethirtyeight')


class Network(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Distribute:  weight matrix init (uniform, normal)
    # Basis_final: ignored when basis is a list

    def __init__(self, units, basis=basis_bent, distribute=dist_normal, basis_final=None):

        # Weight and bias initialization.
        # Initial random numbers are scaled by layer size (reminds me of standard error of the mean)
        self.weights = []
        for idx in range(len(units) - 1):
            self.weights.append(Array(distribute(units[idx + 1], units[idx] + 1) / np.sqrt(units[idx])))

        # Broadcast basis function, so that each layer has one
        if type(basis) is not list:
            basis = [basis] * len(units)

            # Make it easy to specify the final layer when broadcasting
            if basis_final is not None:
                basis[-1] = basis_final
        self.basis = basis

    def predict(self, data):
        """Stimulus evaluation"""

        for idx in range(len(self.weights)):
            bias = np.ones([1, data.shape[1]])
            #  r = basis          (W                 * s)
            data = self.basis[idx](self.weights[idx] @ np.vstack([data, bias]))
        return data

    def save(self, name='network'):
        np.savez('./data/trained/' + name, self.weights)

        with open('./data/trained/' + name, 'w') as outfile:
            json.dump({'bases': [str(b) for b in self.basis]}, outfile)

    def load(self, name='network'):
        self.weights = np.load('./data/trained' + name + '.npz')
