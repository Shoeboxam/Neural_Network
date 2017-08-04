import matplotlib.pyplot as plt
import os

plt.style.use('fivethirtyeight')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        prediction = self.network.predict(inputs)
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

    @staticmethod
    def _unbroadcast(parameter):
        if type(parameter) is list:
            print('TF wrapper does not support vectors of hyperparameters')
            return parameter[0]
        return parameter