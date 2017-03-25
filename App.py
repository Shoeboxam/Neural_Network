import numpy as np
from Neural import Neural
from Function import *
from Environment import *

np.set_printoptions(suppress=True)

# env = Logic_Gate(np.array([[0], [1], [1], [0]]))
# env = Logic_Gate(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))
# env = Logic_Gate(np.array([[1], [0], [0], [1], [1], [0], [1], [0]]))
env = Continuous(lambda v: ((v-.1)**2 + .25), bounds=[-1, 1])

# ~~~~Learning machine parameters~~~~
layers = [env.shape_input()[1], 12, 23, 15, env.shape_output()[1]]     # Number of nodes per layer

# Create the net
net = Neural(layers, delta=delta_linear, basis=basis_relu, gamma=[.05, .05, .7, .1], debug=True)
net.train(env)
print(net.evaluate(env.survey()[0]))
