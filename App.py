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
params = {
    "units": [env.shape_input()[1], 23, 20, env.shape_output()[1]],  # shape of network
    "basis": basis_bent,  # choice of bases

    # Learning rate function
    "learn_step": .0001,
    "learn": learn_power,

    # Weight decay regularization function
    "decay_step": 0.0001,
    "decay": decay_NONE,

    # Momentum preservation
    "moment_step": .1,

    # Percent of weights to drop each training iteration
    "dropout": 0.2,

    # Error function
    "delta": delta_sum_squared,

    "epsilon": 0.1,        # error allowance
    "iterations": 5000,    # limit on number of iterations to run
    "debug": True,         # plot graphs
    }

# Create the net
net = Neural(**params)

# Train the net
net.train(env)

# Test the net
print(net.propagate(env.survey()[0].T))
