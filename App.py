from Function import *
from Neural import Neural

from Environment import *

np.set_printoptions(suppress=True)

# Select an environment
env = MNIST()
# env = Continuous(lambda v: (24 * v**4 - 2 * v**2 + v), bounds=[-1, 1])
# env = Logic_Gate(np.array([[0], [1], [1], [0], [1], [0], [0], [0]]))
# env = Logic_Gate(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))
# env = Logic_Gate(np.array([[1], [0], [0], [1], [0], [0], [1], [0]]))

# ~~~ Create the network ~~~
init_params = {
    # Shape of network
    "units": [env.size_input(), 20, env.size_output()],

    # Basis function(s) from Function.py
    "basis": [basis_bent, basis_softmax],

    # Error function from Function.py
    "delta": delta_cross_entropy
    }

net = Neural(**init_params)

# ~~~ Train the network ~~~
train_params = {
    # Source of stimuli
    "environment": env,

    # Learning rate function
    "learn_step": .5,
    "learn": learn_fixed,

    # Weight decay regularization function
    "decay_step": 0.0001,
    "decay": decay_NONE,

    # Momentum preservation
    "moment_step": 0,

    # Percent of weights to drop each training iteration
    "dropout": 0,

    "epsilon": .04,           # error allowance
    "iteration_limit": 500000,  # limit on number of iterations to run

    "debug": True,
    "graph": False
    }

net.train(**train_params)

# ~~~ Test the network ~~~
[stimuli, expectation] = env.survey()
print(net.predict(stimuli.T))