import numpy as np
from .Function import *
# The network graph is constructed of gates that expose a forward pass and a backward pass.


class Gate(object):
    def __init__(self, children):

        # Remember children
        if type(children) is not list:
            children = [children]
        self.children = children

        # Store link to parent in child
        for child in children:
            child.parents.append(self.__class__)
        self.parents = []

        self._iteration = 0
        self._forward_cache = []
        self._backward_cache = []


class Transform(Gate):
    name = 'transform'

    def __init__(self, children, nodes, distribute=dist_normal):
        super().__init__(children)

        # Number of output nodes
        if type(nodes) is not list:
            nodes = [nodes]
        self.output_nodes = nodes
        self.input_nodes = [self.node_search(child) for child in self.children]

        # Initialize weight matrix
        self.weights = distribute(sum(self.output_nodes), sum(self.input_nodes) + 1) * np.sqrt(2 / self.nodes)

        self.output_slices = []
        cursor = 0
        for sibling_id in range(len(self.output_nodes)):
            space = self.output_nodes[sibling_id]
            self.output_slices.append(slice(cursor, cursor + space))
            cursor += space

        self.input_slices = []
        cursor = 0
        for sibling_id in range(len(self.input_nodes)):
            space = self.input_nodes[sibling_id]
            self.input_slices.append(slice(cursor, cursor + space))
            cursor += space

    def __call__(self, caller, i, stimulus, deriv=None):
        parent_id = self.parents.index(caller)

        if self._iteration == i:
            if not deriv:
                return self._forward_cache
            return self._backward_cache

        if not deriv:
            stimulus = np.vstack([child(stimulus) for child in self.children])
            self._forward_cache = self.weights @ stimulus
            return self._forward_cache[self.output_slices[parent_id], -1]
        else:
            return self.weights[self.output_slices[parent_id], -1]

    @staticmethod
    def node_search(children):
        """Count the number of input nodes"""
        node_count = 0
        for child in children:
            if child.name is 'transform':
                node_count += child.nodes
            elif child.name is 'stimulus':
                node_count += child.nodes
            else:
                node_count += node_search(child.children)
        return node_count


class Softplus(Gate):
    name = 'Softplus'

    def __call__(self, i, stimulus, d=0):
        return np.log(1 + np.exp(stimulus))
