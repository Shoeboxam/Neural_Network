from Function import *
from Neural import Neural

from Environment import *

np.set_printoptions(suppress=True)

# Select an environment
env = Continuous(lambda v: (24 * v**4 - 2 * v**2 + v), bounds=[-1, 1])
# env = Logic_Gate(np.array([[0], [1], [1], [0], [1], [0], [0], [0]]))
# env = Logic_Gate(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))
# env = Logic_Gate(np.array([[1], [0], [0], [1], [0], [0], [1], [0]]))

# ~~~ Create the network ~~~
init_params = {
    # Shape of network
    "units": [env.shape_input()[1], 23, 20, env.shape_output()[1]],

    # Basis function(s) from Function.py
    "basis": basis_bent,

    # Error function from Function.py
    "delta": delta_sum_squared
    }

net = Neural(**init_params)

# ~~~ Train the network ~~~
train_params = {
    # Source of stimuli
    "environment": env,

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

    "epsilon": 0.1,           # error allowance
    "iteration_limit": 5000,  # limit on number of iterations to run
    "debug": True             # plot graphs
    }

net.train(**train_params)

# ~~~ Test the network ~~~
[stimuli, expectation] = env.survey()
print(net.predict(stimuli.T))