from Function import *
from Neural import Neural

from Environment import *

np.set_printoptions(suppress=True)

# ~~~~Learning machine environments~~~~
env = Continuous(lambda v: (24 * v**4 - 2 * v**2 + v), bounds=[-1, 1])
# env = Logic_Gate(np.array([[0], [1], [1], [0], [1], [0], [0], [0]]))
# env = Logic_Gate(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))
# env = Logic_Gate(np.array([[1], [0], [0], [1], [0], [0], [1], [0]]))

# ~~~~Learning machine parameters~~~~
params = {"units":   [env.shape_input()[1], 23, 20, env.shape_output()[1]],  # shape of network
          "basis":   [basis_bent, basis_bent, basis_bent],                   # choice of bases
          "gamma":   [.01, .01, .01],    # step sizes for each layer
          "regul":   reg_NONE,           # choice of regularization method
          "delta":   delta_linear,       # choice of error function
          "epsilon": .1,                 # error allowance
          "debug":   True                # plot graphs
          }

# Create the net
net = Neural(**params)

# Train the net
net.train(env)

# Test the net
print(net.evaluate(env.survey()[0]))
