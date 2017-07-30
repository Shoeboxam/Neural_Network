import matplotlib.pyplot as plt

from .Function import *
from .Array import Array
from .Optimizer import Optimizer

plt.style.use('fivethirtyeight')


class Neural_Network(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Delta:       sum squared, cross entropy error

    def __init__(self, units, basis=basis_bent, distribute=dist_normal):

        # Weight and bias initialization. Initial random numbers are scaled by layer size.
        self.weights = []
        self.biases = []
        for idx in range(len(units) - 1):
            self.weights.append(Array(distribute(units[idx + 1], units[idx]) / np.sqrt(units[idx])))
            self.biases.append(Array(np.zeros((units[idx + 1], 1))))

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)
        self.basis = basis

    def predict(self, data):
        """Stimulus evaluation"""

        for idx in range(len(self.weights)):
            #  r = basis          (W                 * x    + b)
            data = self.basis[idx](self.weights[idx] @ data + self.biases[idx])
        return data

    # Environment: class with a 'sample stimulus' method
    # Batch size:  number of samples per training epoch
    # Optimizer:   dictionary of optimizer initialization settings
    # Learn step:  learning parameter
    # Learn:       learning function
    # Decay step:  weight decay parameter
    # Decay:       weight decay function
    # Dropout:     percent of nodes to drop from each layer
    # Epsilon:     convergence allowance
    # Iterations:  number of iterations to run. If none, then no limit
    # Debug:       make graphs and log progress to console
    # Convergence: grad, newt *not implemented

    def train(self, environment, batch_size=1,
              optimizer=None,
              cost=cost_sum_squared,
              learn_step=1e-2, anneal=anneal_fixed,
              decay_step=None, decay=decay_NONE, dropout=0,
              epsilon=1e-2, iteration_limit=None,
              debug=False, graph=False):

        # --- Setup parameters ---

        if optimizer is None:
            optimizer = {'method': 'gradient_descent'}

        # Create optimizers
        weight_optimizer = []
        bias_optimizer = []
        for idx in range(len(self.weights)):

            # Shapes are needed to initialize history variables
            optimizer['shape'] = self.biases[idx].shape
            bias_optimizer.append(Optimizer(**optimizer))

            optimizer['shape'] = self.weights[idx].shape
            weight_optimizer.append(Optimizer(**optimizer))

        # Learning parameters
        if type(learn_step) is float or type(learn_step) is int:
            learn_step = [learn_step] * len(self.weights)

        # Decay parameters
        if decay_step is None:
            decay_step = learn_step

        if type(decay_step) is float or type(decay_step) is int:
            decay_step = [decay_step] * len(self.weights)

        # --- Define propagation within net ---

        # Internal variables to reduce time complexity of n layers in training deep nets
        cache_iteration = 0
        cache_weights = []

        def propagate(data, depth=None, cache=False):
            """Stimulus evaluation with support for configurable depth, caching and dropout."""
            # Depth can limit evaluation to a certain number of layers in the net

            nonlocal cache_iteration
            nonlocal cache_weights

            if depth is None:
                depth = len(self.weights)

            root = 0

            # Early return if weight set already computed
            if cache:
                # Check for validity of cached weights
                if iteration == cache_iteration:

                    # Check if cache has been computed to necessary depth
                    if depth < len(cache_weights):
                        return cache_weights[depth]
                    else:
                        root = len(cache_weights) - 1
                        data = cache_iteration[root]

                else:
                    # Cache has been invalidated
                    cache_weights = []

            for layer_id in range(root, depth):

                # Dropout
                if dropout:
                    # Drop nodes from network
                    data *= np.random.binomial(1, (1.0 - dropout), size=data.shape)
                    # Resize remaining nodes to compensate for loss of nodes
                    data = data.astype(float) * (1.0 / (1 - dropout))
                data = self.weights[layer_id] @ data + self.biases[layer_id]
                data = self.basis[layer_id](data)

                if cache:
                    cache_weights.append(data)

            return data

        # --- Actual training portion ---
        iteration = 0
        pts = []

        converged = False
        while not converged:
            iteration += 1
            # print("Iteration: " + str(iteration))

            # Choose a stimulus
            stimulus, expect = map(Array, environment.sample(quantity=batch_size))

            # Layer derivative accumulator
            dq_dq = Array(np.eye(self.weights[-1].shape[0]))

            # Loss function derivative
            dln_dq = cost(expect, propagate(stimulus, cache=True), d=1)[np.newaxis, ...]

            # Train each weight set sequentially
            for layer in reversed(range(len(self.weights))):

                # ~~~~~~~ Loss derivative phase ~~~~~~~
                # stimulus = value of previous basis function or input stimulus
                s = propagate(stimulus, depth=layer, cache=True)

                # reinforcement = W     x s + b
                r = self.weights[layer] @ s + self.biases[layer]

                # Basis function derivative
                dq_dr = Array(self.basis[layer](r, d=1))

                # Reinforcement function derivative
                dr_dWvec_i = []
                for feature in s.T:
                    dr_dWvec_i.append(np.kron(feature, np.identity(self.weights[layer].shape[0])))
                dr_dWvec = Array(np.dstack(dr_dWvec_i))

                # Chain rule for full derivative
                dln_dWvec = dln_dq @ dq_dq @ dq_dr @ dr_dWvec
                dln_db = dln_dq @ dq_dq @ dq_dr  # @ dr_db (Identity matrix)

                # Unvectorize
                dln_dW = np.reshape(dln_dWvec.T, [*self.weights[layer].shape, batch_size])

                # ~~~~~~~ Gradient descent phase ~~~~~~~

                learn_rate = anneal(iteration, iteration_limit) * learn_step[layer]
                weight_decay = decay_step[layer] * decay(self.weights[layer], d=1)

                # Take a step towards the minima
                self.biases[layer] += bias_optimizer[layer](learn_rate, np.average(dln_db, axis=2).T)
                self.weights[layer] += (weight_optimizer[layer](learn_rate, np.average(dln_dW, axis=2)) + weight_decay)

                # ~~~~~~~ Update internal state ~~~~~~~
                # Store derivative accumulation for next layer
                dr_dq = self.weights[layer]
                dq_dq = dq_dq @ dq_dr @ dr_dq

                # Check for oversaturated weights
                if debug:
                    maximum = max(self.weights[layer].min(), self.weights[layer].max(), key=abs)
                    if maximum > 1000:
                        print("Layer " + str(layer) + " weights are too large: " + str(maximum))

            # --- Debugging and graphing ---
            # Exit condition
            if iteration_limit is not None and iteration >= iteration_limit:
                break

            if (graph or epsilon or debug) and iteration % 50 == 0:
                [inputs, expectation] = map(Array, environment.survey())
                prediction = self.predict(inputs)
                error = environment.error(expectation, prediction)

                if error < epsilon:
                    converged = True

                if debug:
                    print("Error: " + str(error))
                    # print(expectation)
                    # print(prediction)

                if graph:
                    pts.append((iteration, error))

                    plt.subplot(1, 2, 1)
                    plt.cla()
                    plt.title('Error')
                    plt.plot(*zip(*pts), marker='.', color=(.9148, .604, .0945))
                    plt.pause(0.00001)

                    plt.subplot(1, 2, 2)
                    plt.cla()
                    plt.title('Environment')

                    # Default graphing behaviour defined in environment.py
                    environment.plot(plt, prediction)

                    plt.pause(0.00001)
