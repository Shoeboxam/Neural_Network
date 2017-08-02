import matplotlib.pyplot as plt
from ..Array import Array

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
        plt.pause(0.00001)

        # Environment plot
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.title('Environment')
        self.environment.plot(plt, self.plot_points, prediction)

        plt.pause(0.00001)

    def _broadcast(self, parameter):
        if type(parameter) is not list:
            return Array([parameter] * len(self.network.weights))
        if len(parameter) != len(self.network.weights):
            raise ValueError('Parameter length does not match number of layers.')
        return Array(parameter)