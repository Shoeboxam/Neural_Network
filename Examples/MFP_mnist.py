# Use custom implementation:
from MFP import *

# Use MFP_TF wrapper:
# from MFP_TF import *

import urllib.request
from io import BytesIO
import os
import gzip
import struct

import numpy as np
np.set_printoptions(suppress=True)


class MNIST:
    def __init__(self):
        source = 'http://yann.lecun.com/exdb/mnist'
        target = os.path.abspath('./data/MNIST/')

        datasets = {
            'train_images': 'train-images.idx3-ubyte',
            'train_labels': 'train-labels.idx1-ubyte',
            'test_images': 't10k-images.idx3-ubyte',
            'test_labels': 't10k-labels.idx1-ubyte'
        }

        if not os.path.exists(target):
            os.makedirs(target)

            datasets_compressed = {
                'train_images': 'train-images-idx3-ubyte.gz',
                'train_labels': 'train-labels-idx1-ubyte.gz',
                'test_images': 't10k-images-idx3-ubyte.gz',
                'test_labels': 't10k-labels-idx1-ubyte.gz'
            }

            for ident, filename in datasets_compressed.items():
                compressed_file = BytesIO()
                compressed_file.write(urllib.request.urlopen(source + '/' + filename).read())
                compressed_file.seek(0)

                decompressed_file = gzip.GzipFile(fileobj=compressed_file, mode='rb')

                with open(target + '\\' + datasets[ident], 'wb') as outfile:
                    outfile.write(decompressed_file.read())

        self.train_images = MNIST._load_images(target + '/' + datasets['train_images'])
        self.train_labels = MNIST._load_labels(target + '/' + datasets['train_labels'])

        self.test_images = MNIST._load_images(target + '/' + datasets['test_images'])
        self.test_labels = MNIST._load_labels(target + '/' + datasets['test_labels'])

    @staticmethod
    def _load_images(file_name):
        with open(file_name, 'rb') as image_file:
            ftype, length, rows, cols = struct.unpack(">IIII", image_file.read(16))
            return np.fromfile(image_file, dtype=np.uint8).reshape(length, rows * cols).astype(float) / 128 - 1

    @staticmethod
    def _load_labels(file_name):
        with open(file_name, 'rb') as label_file:
            label_file.read(8)  # Ignore header info for file type and number of entries
            integer_matrix = np.fromfile(label_file, dtype=np.int8)

            # Convert to one-hot format
            bool_matrix = np.zeros((np.size(integer_matrix), 10))
            bool_matrix[np.arange(np.size(integer_matrix)), integer_matrix] = True
            return bool_matrix

    def sample(self, quantity=1):
        x = np.random.randint(np.size(self.train_images[0]), size=quantity)
        return [self.train_images[x].T, self.train_labels[x].T]

    def survey(self, quantity=200):
        if quantity is None:
            return self.test_images.T, self.test_labels.T
        else:
            x = np.linspace(0, np.size(self.test_images[0]), quantity).astype(int)  # Size changes error granularity
            return [self.test_images[x].T, self.test_labels[x].T]

    def size_input(self):
        return np.size(self.train_images[0])

    def size_output(self):
        return np.size(self.train_labels[0])

    def plot(self, plt, predict):
        # Do not attempt to plot an image
        pass

    @staticmethod
    def error(expect, predict):
        predict_id = np.argmax(predict, axis=0)
        expect_id = np.argmax(expect, axis=0)
        return int((1.0 - np.mean((predict_id == expect_id).astype(float))) * 100)


environment = MNIST()

# ~~~ Create the network ~~~
network_params = {
    # Shape of network
    "units": [environment.size_input(), 100, 100, 50, environment.size_output()],

    # Basis function(s) from network's Function.py
    "basis": basis_bent,
    "basis_final": basis_softmax,

    # Distribution to use for weight initialization
    "distribute": dist_normal
    }

network = Network(**network_params)

# ~~~ Train the network ~~~
optimizer_params = {
    # Source of stimuli
    "batch_size": 5,

    "cost": cost_cross_entropy,

    # Learning rate
    "learn_step": 0.5,
    "learn_anneal": anneal_fixed,

    # Weight decay regularization function
    "regularize_step": 0.0,
    "regularizer": reg_L2,

    "epsilon": 0.0,           # error allowance
    "iteration_limit": None,  # limit on number of iterations to run

    "debug_frequency": 1000,
    "debug": True,

    # The error measurement used in the mnist graph is highly susceptible to sampling error
    "graph": False
    }

Adagrad(network, environment, **optimizer_params).minimize()

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.survey()
print(network.predict(stimuli))
