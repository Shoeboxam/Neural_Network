# Learn a continuous function
from inspect import signature

# Use custom implementation:
# from Jacobian_Chain import *

# Use Tensorflow wrapper:
from Tensorflow_Wrapper import *


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.set_printoptions(suppress=True, linewidth=10000)


class Continuous:

    def __init__(self, funct, domain, range=None):

        self._size_input = len(signature(funct[0]).parameters)
        self._size_output = len(funct)

        self._funct = funct
        self._domain = domain

        if range is None:
            self._range = [[-1, 1]] * len(funct)

            if self._size_input == 1 and self._size_output == 1:
                candidates = self._funct[0](np.linspace(*self._domain[0], num=100))
                self._range = [[min(candidates), max(candidates)]]
        else:
            self._range = range

        self.viewpoint = np.random.randint(0, 360)

    def sample(self, quantity=1):
        # Generate random values for each input stimulus
        axes = []
        for idx in range(self._size_input):
            axes.append(np.random.uniform(low=self._domain[idx][0], high=self._domain[idx][1], size=quantity))

        stimulus = np.array(np.meshgrid(*axes)).reshape(self._size_input, -1)

        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    def survey(self, quantity=100):
        # Generate random values for each input stimulus
        axes = []
        for idx in range(self._size_input):
            axes.append(np.linspace(start=self._domain[idx][0], stop=self._domain[idx][1], num=quantity))

        stimulus = np.array(np.meshgrid(*axes)).reshape(self._size_input, -1)
        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    def size_input(self):
        return self._size_input

    def size_output(self):
        return self._size_output

    def plot(self, plt, predict):
        x, y = self.survey()

        # Output of function is 1 dimensional
        if y.shape[0] == 1:
            ax = plt.subplot(1, 2, 2)
            plt.ylim(self._range[0])

            ax.plot(x[0], y[0], marker='.', color=(0.3559, 0.7196, 0.8637))
            ax.plot(x[0], predict[0], marker='.', color=(.9148, .604, .0945))

        # Output of function has arbitrary dimensions
        if y.shape[0] > 1:
            ax = plt.subplot(1, 2, 2, projection='3d')
            ax.plot(x[0], y[0], y[1], color=(0.3559, 0.7196, 0.8637))
            ax.plot(x[0], predict[0], predict[1], color=(.9148, .604, .0945))
            ax.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 5

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)


# environment = Continuous([lambda a, b: (24 * a**4 - 2 * b**2 + a),
#                           lambda a, b: (-5 * a**3 + 2 * b**2 + b),
#                           lambda a, b: (12 * a**2 + 8 * b**3 + b),
#                           lambda a, b: (12 * a ** 2 + 12 * b ** 3 + b)], domain=[[-1, 1]] * 2)

# environment = Continuous([lambda a, b: (24 * a - 2 * b**2 + a),
#                           lambda a, b: (-5 * a**3 + 2 * b**2 + b)], domain=[[-1, 1]] * 2)

environment = Continuous([lambda x: np.sin(x),
                          lambda x: np.cos(x)], domain=[[-2 * np.pi, 10 * np.pi], [-np.pi, np.pi]])

# environment = Continuous([lambda a: (24 * a**2 + a),
#                           lambda a: (-5 * a**3)], domain=[[-1, 1]])

# environment = Continuous([lambda v: (24 * v**4 - 2 * v**2 + v)], domain=[[-1, 1]])

# ~~~ Create the network ~~~
network_params = {
    # Shape of network
    "units": [environment.size_input(), 5, environment.size_output()],

    # Basis function(s) from Optimizer.py
    "basis": basis_sinusoid,

    # Weight initialization distribution
    "distribute": dist_normal
    }

network = Network(**network_params)

# ~~~ Train the network ~~~
optimizer_params = {
    # Source of stimuli
    "batch_size": 10,

    # Error function from Optimizer.py
    "cost": cost_sum_squared,

    # Learning rate
    "learn_step": .1,
    "anneal": anneal_inverse,

    # Weight decay regularization function
    "regularize_step": 0.0,
    "regularizer": reg_L12,

    # Percent of weights to drop each training iteration
    # "dropout_step": 0.05,
    # "dropconnect_step": 0.05,

    "epsilon": .04,           # error allowance
    "iteration_limit": 500000,  # limit on number of iterations to run

    "debug_frequency": 50,
    "debug": True,
    "graph": True
    }

Adagrad(network, environment, **optimizer_params).minimize()

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.survey()
print(network.predict(stimuli))
