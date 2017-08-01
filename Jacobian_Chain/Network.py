import matplotlib.pyplot as plt
from .Array import Array
from .Function import *


plt.style.use('fivethirtyeight')


class Network(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Distribute:  weight matrix init (uniform, normal)

    def __init__(self, units, basis=basis_bent, distribute=dist_normal):

        # Weight and bias initialization.
        # Initial random numbers are scaled by layer size (reminds me of standard error of the mean)
        self.weights = []
        for idx in range(len(units) - 1):
            self.weights.append(Array(distribute(units[idx + 1], units[idx] + 1) / np.sqrt(units[idx])))

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)
        self.basis = basis

    def predict(self, data):
        """Stimulus evaluation"""

        for idx in range(len(self.weights)):
            bias = np.ones([1, data.shape[1]])
            #  r = basis          (W                 * s)
            data = self.basis[idx](self.weights[idx] @ np.vstack([data, bias]))
        return data
