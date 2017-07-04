# Template for how to represent datasets to the network


class Environment:

    def sample(self):
        # Return a single element and label from the event environment
        pass

    def survey(self):
        # Return collection of elements and labels from the event environment
        pass

    def size_input(self):
        # Number of input nodes
        pass

    def size_output(self):
        # Number of output nodes
        pass

    def range(self):
        # Optional: return expected upper and lower bounds of output, useful for graphing
        pass

    def plot(self, plt, predict):
        # Default method to graph a plot
        plt.ylim(self._range)
        x, y = self.survey()
        plt.plot(x, y, marker='.', color=(0.3559, 0.7196, 0.8637))
        plt.plot(x, predict.T[0], marker='.', color=(.9148, .604, .0945))

    @staticmethod
    def error(expect, predict):
        # How to quantify error
        pass
