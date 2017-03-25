import numpy as np
from Neural import Neural
from Function import *
from Environment import *

np.set_printoptions(suppress=True)

# environment = Logic_Gate(np.array([[0], [1], [1], [0]]))
environment = Logic_Gate(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))
# environment = Logic_Gate(np.array([[1], [0], [0], [1], [1], [0], [1], [0]]))

# ~~~~Learning machine parameters~~~~
layers = [environment.shape_input()[1], 12, 23, 15, environment.shape_output()[1]]     # Number of nodes per layer

# Create the net
net = Neural(layers, delta=delta_linear, basis=basis_relu, gamma=[.05, .05, .7, .1], debug=True)
net.train(environment)
print(net.evaluate(environment.survey()[0]))
