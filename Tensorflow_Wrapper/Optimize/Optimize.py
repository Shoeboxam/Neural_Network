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

        if self.graph:
            self.plot_points = []

    def convergence_check(self, **kwargs):
        [stimuli, expectation] = self.environment.survey()
        prediction = self.network.predict(stimuli)
        error = self.environment.error(expectation, prediction)

        if error < self.epsilon:
            return True

        kwargs = {**{'error': error,
                     'prediction': prediction
                     }, **kwargs}

        if self.debug:
            self.post_debug(**kwargs)

        if self.graph:
            self.post_graph(**kwargs)

        return False

    def post_debug(self, **kwargs):
        print("Iteration: " + str(kwargs['iteration']))
        print("Error: " + str(kwargs['error']))

        # print(kwargs['prediction'])
        # print(kwargs['expectation'])
        pass

    def post_graph(self, **kwargs):
        self.plot_points.append((kwargs['iteration'], kwargs['error']))

        # Error plot
        plt.subplot(1, 2, 1)
        plt.title('Error')
        plt.plot(*zip(*self.plot_points), marker='.', color=(.9148, .604, .0945))
        plt.pause(0.00001)

        # Environment plot
        plt.subplot(1, 2, 2)
        plt.title('Environment')
        self.environment.plot(plt, kwargs['prediction'])

        plt.pause(0.00001)

    @staticmethod
    def _unbroadcast(parameter):
        if type(parameter) is list:
            print('TF wrapper does not support vectors of hyperparameters')
            return parameter[0]
        return parameter
