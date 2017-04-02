from Function import *
from Neural import Neural

from Feed_Forward.Environment import *

np.set_printoptions(suppress=True)

# ~~~~Learning machine environments~~~~
env = Continuous(lambda v: (24 * v**4 - 2 * v**2 + v), bounds=[-1, 1])
# env = Logic_Gate(np.array([[0], [1], [1], [0]]))
# env = Logic_Gate(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))
# env = Logic_Gate(np.array([[1], [1], [0], [1], [1], [0], [0], [0]]))

# ~~~~Learning machine parameters~~~~
layers = [env.shape_input()[1], 23, 20, env.shape_output()[1]]     # Number of nodes per layer
# Create the net
net = Neural(layers, delta=delta_linear, gamma=[.01, .01, .01], debug=True)
net.train(env)
print(net.evaluate(env.survey()[0]))
