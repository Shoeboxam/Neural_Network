import itertools
import math
import numpy as np

import urllib.request
from io import BytesIO
import os
import gzip
import struct


class Environment(object):

    def sample(self):
        # Return a single element and label from the event environment
        pass

    def survey(self):
        # Return collection of elements and labels from the event environment
        pass

    def range(self):
        # Return expected upper and lower bounds of output, useful for graphing
        pass

    def size_input(self):
        # Number of input nodes
        pass

    def size_output(self):
        # Number of output nodes
        pass


class Logic_Gate(Environment):

    def __init__(self, expectation):
        bit_length = math.log(np.shape(expectation)[0], 2)
        if bit_length % 1 != 0:
            raise TypeError('Length of expectation must be a power of two.')

        self._expectation = expectation
        self._environment = np.array([i for i in itertools.product([0, 1], repeat=int(bit_length))])

    def sample(self):
        choice = np.random.randint(np.shape(self._environment)[0])
        return self._environment[choice], self._expectation[choice]

    def survey(self):
        return [self._environment, self._expectation]

    def range(self):
        return [0, 1]

    def size_input(self):
        return np.shape(self._environment)[1]

    def size_output(self):
        return np.shape(self._expectation)[0]


class Continuous(Environment):

    def __init__(self, funct, bounds):
        self._funct = np.vectorize(funct)
        self._bounds = bounds

        candidates = self._funct(np.linspace(*self._bounds, num=100))
        self._range = [min(candidates), max(candidates)]

    def sample(self):
        x = np.random.uniform(*self._bounds)
        return [[x], [self._funct(x)]]

    def survey(self):
        x = np.linspace(*self._bounds, num=100)
        return [np.vstack(x), self._funct(x)]

    def range(self):
        return self._range

    def size_input(self):
        return 1

    def size_output(self):
        return 1


class MNIST(Environment):
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
            return np.fromfile(image_file, dtype=np.uint8).reshape(length, rows * cols).astype(float) / 256

    @staticmethod
    def _load_labels(file_name):
        with open(file_name, 'rb') as label_file:
            label_file.read(8)  # Ignore header info for file type and number of entries
            return np.fromfile(label_file, dtype=np.int8)

    def sample(self):
        x = np.random.randint(np.size(self.train_images[0]))
        return [self.train_images[x], self.train_labels[x]]

    def survey(self):
        x = np.random.randint(np.size(self.test_images[0]), size=20)
        return [self.test_images[x], self.test_labels[x]]

    def range(self):
        return [0, 1]

    def size_input(self):
        return np.size(self.train_images[0])

    def size_output(self):
        return np.size(self.train_labels[0])
