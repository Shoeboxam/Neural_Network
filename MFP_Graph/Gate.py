import numpy as np


class Gate(object):
    def __init__(self, children):
        if type(children) is not list:
            children = [children]
        self.children = children

    def __call__(self, stimulus):
        return np.vstack([child(stimulus) for child in self.children])

    def gradient(self, variable):
        return np.vstack([child.gradient(variable) for child in self.children])

    @property
    def output_nodes(self):
        return 0

    @property
    def input_nodes(self):
        """Count the number of input nodes"""
        node_count = 0
        for child in self.children:
            if child.__class__.__name__ in ['Transform', 'Stimulus']:
                node_count += child.output_nodes
            else:
                node_count += child.input_nodes
        return node_count


class Transform(Gate):
    def __init__(self, children, output_nodes):
        super().__init__(children)
        self.weights = np.random.normal(size=(output_nodes, self.input_nodes))

    @property
    def output_nodes(self):
        return self.weights.shape[0]


class Softplus(Gate):
    def __call__(self, stimulus):
        features = super()(stimulus)
        return 1.0 / (1.0 + np.exp(-features))
