# Core data structure for a Restricted Boltzmann Machine

import numpy as np
from .Function import *


class RBM(object):
    def __init__(self, input_nodes, output_nodes, basis=basis_logistic, distribute=dist_normal):
        # Added one bias node that is fully connected to both the input and output layers
        self.weight = distribute(output_nodes + 1, input_nodes + 1)
        print(self.weight)

        # Either logistic or softmax
        self.basis = basis

    def predict(self, data):
        # stimuli are always given a row of ones to represent the bias/constant
        bias = np.ones([1, data.shape[1]])

        # probability =      basis(W           * s)
        probabilities = self.basis(self.weight @ np.vstack([data, bias]))

        return (probabilities > np.random.rand(*probabilities.shape))[:1, :]
