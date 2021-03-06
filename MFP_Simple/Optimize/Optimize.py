# Optimizer foundation
# Handles graphing, debug and settings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('fivethirtyeight')


class Optimize(object):
    def __init__(self, network, environment, **kwargs):
        self.network = network
        self.environment = environment

        preferences = {**{
            "epsilon": None,           # error allowance
            "iteration_limit": 10000,  # limit on number of iterations to run

            "debug_frequency": 50,
            "debug": False,
            "graph": False
        }, **kwargs}

        for key, value in preferences.items():
            setattr(self, key, value)

        self.iteration = 0

        if self.graph:
            self.plot_points = []

    def convergence_check(self):
        [inputs, expectation] = self.environment.survey()
        prediction = []
        for stim in inputs.T:
            prediction.append(self.network.predict(stim))

        prediction = np.array(prediction).T
        error = self.environment.error(expectation, prediction)

        if error < self.epsilon:
            return True

        if self.debug:
            self.post_debug(error=error, prediction=prediction)

        if self.graph:
            self.post_graph(error, prediction)

        return False

    def post_debug(self, **kwargs):
        print("Iteration: " + str(self.iteration))
        print("Error: " + str(kwargs['error']))

        for layer, weight in enumerate(self.network.weights):
            # Check for oversaturated weights
            if self.debug:
                maximum = max(weight.min(), weight.max(), key=abs)
                if maximum > 1000:
                    print("Layer " + str(layer) + " weights are too large: " + str(maximum))

        # print(kwargs['prediction'])
        # print(kwargs['expectation'])

    def post_graph(self, error, prediction):
        self.plot_points.append((self.iteration, error))

        # Error plot
        plt.subplot(1, 2, 1)
        plt.cla()
        plt.title('Error')
        plt.plot(*zip(*self.plot_points), marker='.', color=(.9148, .604, .0945))

        # Environment plot
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.title('Environment')
        self.environment.plot(plt, prediction)

        plt.pause(0.00001)

    def _broadcast(self, parameter):
        if type(parameter) is not list:
            return [parameter] * len(self.network.weights)
        return parameter
