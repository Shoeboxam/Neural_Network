import numpy as np
from .Variable import Variable


# Return cached value if already computed in current iteration
def _cache(method):
    def decorator(self, **kwargs):
        if kwargs['i'] and self._iteration == kwargs['i']:
            return lambda **kw: getattr(self, '_cached_' + method.__name__)
        feature = method(self, **kwargs)

        if kwargs['i']:
            setattr(self, '_cached_' + method.__name__, feature)
            setattr(self, '_iteration', kwargs['i'])
        return feature

    return decorator


class Gate(object):
    def __init__(self, children):
        if type(children) is not list:
            children = [children]
        self.children = children

        # Stores references to variables in the gate
        self._variables = {}

        self._iteration = 0
        self._cached___call__ = None
        self._cached_gradient = {}

    def __call__(self, stimulus, i=None):
        return np.vstack([child(stimulus, self) for child in self.children])

    def gradient(self, grad, stimulus, variable):
        if variable not in self.variables():
            # TODO: This shape may be incorrect
            return grad @ np.zeros(self(stimulus).shape)
        return grad @ np.vstack([child.gradient(stimulus, variable[idx]) for idx, child in enumerate(self.children)])

    @property
    def output_nodes(self):
        return sum([child.output_nodes for child in self.children])

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

    @_cache
    @property
    def variables(self):
        """List the input variables"""
        variables = self._variables.keys()
        for child in self.children:
            variables.extend(child.variables())
        return variables


class Transform(Gate):
    def __init__(self, children, weights, biases):
        super().__init__(children)
        self._variables = {
            'weights': weights,
            'biases': biases}

    @property
    def output_nodes(self):
        return self.weights.shape[0]

    @_cache
    def __call__(self, stimulus, i=None):
        bias = np.ones([1, stimulus.shape[1]])

        propagated = super()(stimulus)
        return np.vstack([propagated, bias]) @ self.weights

    @property
    @_cache
    def gradient(self, grad, stimulus, variable):
        propagated = super()(stimulus)
        if variable is self.weights:
            # Full derivative: This is a tensor product simplification to avoid the use of the kron product
            return grad.T @ propagated
        if variable is self.biases:
            return grad
        return super().gradient(grad @ self.weights, stimulus, variable)


class Logistic(Gate):
    @_cache
    def __call__(self, stimulus, i=None):
        features = super()(stimulus)
        return 1.0 / (1.0 + np.exp(-features))

    @property
    @_cache
    def gradient(self, grad, stimulus, variable):
        return super().gradient(grad * self(stimulus) * (1.0 - self(stimulus)), stimulus, variable)



class Stimulus(Gate):
    def __init__(self, environment):
        super().__init__(self)
        if type(environment) is not list:
            environment = [environment]
        self.environment = environment

    @_cache
    def __call__(self, stimulus, i=None):
        return self.environment[stimulus].sample()

    @property
    def gradient(self, grad, stimulus, variable):
        return np.eye(variable.shape[1])
