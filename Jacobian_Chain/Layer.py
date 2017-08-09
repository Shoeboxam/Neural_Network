import numpy as np
from .Function import *


class Layer(object):
    def __init__(self, children):
        if type(children) is not list:
            children = [children]
        self.children = children
        self._iteration = 0
        self._forward_cache = []
        self._backward_cache = []


class Reinforcement(Layer):
    name = 'Reinforcement'

    def __init__(self, children, nodes, distribute=dist_normal):
        super().__init__(children)

        # Number of output nodes
        if type(nodes) is not list:
            nodes = [nodes]
        self.output_nodes = nodes
        self.input_nodes = [self.node_search(child) for child in self.children]

        # Initialize weight matrix
        self.weights = distribute(sum(self.output_nodes), sum(self.input_nodes) + 1) * np.sqrt(2 / self.nodes)

    def __call__(self, i, stimuli, d=0):
        if self._iteration == i:
            if not d:
                return self._forward_cache
            return self._backward_cache

        if forward:
            stimulus = np.vstack([child(stimuli) for child in self.children])
            return self.weights @ stimulus
        else:
            return self.weights[..., :-1]

    @staticmethod
    def node_search(children):
        """Count the number of input nodes"""
        node_count = 0
        for child in children:
            if child.name is 'reinforcement':
                node_count += child.nodes
            elif child.name is 'stimulus':
                node_count += child.nodes
            else:
                node_count += node_search(child.children)
        return node_count


class Softplus(Layer):
    name = 'Softplus'

    def __call__(self, i, stimulus, d=0):
        return np.log(1 + np.exp(stimulus))
