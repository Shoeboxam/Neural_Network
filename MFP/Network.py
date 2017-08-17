# Core data structure for a multilayer feedforward perceptron network
# This implementation is restricted to 'caterpillar' function graphs
# This implementation shares the same interface as the MFP_TF network

import matplotlib.pyplot as plt
from .Array import Array
from .Function import *
import json


plt.style.use('fivethirtyeight')


class MFP(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Distribute:  weight matrix init (uniform, normal)
    # Basis_final: ignored when basis is a list

    def __init__(self, units, basis=basis_bent, distribute=dist_normal, basis_final=None, orthogonalize=False):

        # Weight and bias initialization.
        self.weights = []
        for idx in range(len(units) - 1):
            weights = distribute(units[idx + 1], units[idx] + 1)

            if orthogonalize:
                # Same implementation as Lasagne; https://arxiv.org/pdf/1312.6120.pdf
                u, _, v = np.linalg.svd(weights, full_matrices=False)
                weights = u if u.shape == weights.shape else v

            else:
                # Initial random numbers are scaled by layer size
                weights *= np.sqrt(2 / units[idx])

            self.weights.append(Array(weights))

        # Batch norm: normalization parameters (default is identity)
        self.deviance = [np.ones(nodes) for nodes in units[1:]]
        self.mean = [np.zeros(nodes) for nodes in units[1:]]
        self.scale = [np.ones(nodes) for nodes in units[1:]]
        self.shift = [np.zeros(nodes) for nodes in units[1:]]

        # Input and output (de)normalization
        self.input_mean = np.zeros(units[0])
        self.input_deviance = np.ones(units[0])
        self.output_mean = np.zeros(units[-1])
        self.output_deviance = np.ones(units[-1])

        # Broadcast basis function, so that each layer has one
        if type(basis) is not list:
            basis = [basis] * len(units)

            # Make it easy to specify the final layer when broadcasting
            if basis_final is not None:
                basis[-1] = basis_final
        self.basis = basis

    def predict(self, data):
        """Stimulus evaluation"""
        # Normalize data on input
        data = (data - self.input_mean[:, None]) / (self.input_deviance[:, None] + 1e-8)
        bias = np.ones([1, data.shape[1]])

        def batch_norm(x, l):
            normalized = (x - self.mean[l][:, None]) / (self.deviance[l][:, None] + 1e-8)
            return self.scale[l][:, None] * normalized + self.shift[l][:, None]

        for idx in range(len(self.weights)):
            #  r = basis                     (W                 * s)
            data = self.basis[idx](batch_norm(self.weights[idx] @ np.vstack([data, bias]), idx))

        # Denormalize on output
        print(data)
        return data * self.output_deviance[:, None] + self.output_mean[:, None]

    def save(self, name='network'):
        np.savez('./data/trained/' + name, self.weights)

        with open('./data/trained/' + name, 'w') as outfile:
            json.dump({'bases': [str(b) for b in self.basis]}, outfile)

    def load(self, name='network'):
        self.weights = np.load('./data/trained' + name + '.npz')
