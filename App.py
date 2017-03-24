import numpy as np
from Neural import Neural
from Function import *

np.set_printoptions(suppress=True)

# ~~~~Learning machine parameters~~~~
layers = [2, 23, 12, 1]       # Number of nodes per layer

S = np.array([[1, 1],       # Stimuli data  (environment)
              [0, 1],
              [1, 0],
              [0, 0]])

O = np.array([0, 1, 1, 0])  # Output data (expectation)

# Create the net
net = Neural(layers, delta=delta_linear, basis=basis_sinc, gamma=[.01, .01, .01], debug=True)

net.train(S, O)
print(net.evaluate(S.T))