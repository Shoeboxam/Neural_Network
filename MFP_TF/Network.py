# Core data structure for a multilayer feedforward perceptron network
# This implementation is restricted to 'caterpillar' function graphs
# This implementation shares the same interface as the MFP network


import matplotlib.pyplot as plt

from .Function import *
import tensorflow as tf


class MFP(object):
    # Units:       List of quantity of nodes per layer
    # Basis:       logistic, rectilinear...
    # Delta:       sum squared, cross entropy error

    def __init__(self, units, basis=basis_logistic, distribute=dist_normal, basis_final=None):

        self.units = units

        # Basis functions
        if type(basis) is not list:
            basis = [basis] * len(units)
        self.basis = basis

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Construct placeholders for the input and expected output variables
            self.stimulus = tf.placeholder(tf.float32, [units[0], None], name='stimulus')
            self.expected = tf.placeholder(tf.float32, [units[-1], None], name='expected')

            self.dropout = tf.placeholder(tf.float32, name='dropout')
            self.dropconnect = tf.placeholder(tf.float32, name='dropconnect')

            # Start the hierarchy
            self.hierarchy = self.stimulus
            self.hierarchy_train = self.stimulus

            # Construct and allocate variables that define the network
            for idx in range(len(units) - 1):
                weight = tf.Variable(distribute([units[idx + 1], units[idx]]), name="weight_" + str(idx))
                self.graph.add_to_collection('weights', weight)

                bias = tf.Variable(tf.zeros((units[idx + 1])), name="bias_" + str(idx))
                self.graph.add_to_collection('biases', bias)

                self.hierarchy = basis[idx](weight @ self.hierarchy + bias[..., None])

                # The training hierarchy includes calls for dropout and dropconnect
                weight_dropconnect = tf.nn.dropout(weight, self.dropconnect)
                self.hierarchy_train = tf.nn.dropout(
                    basis[idx](weight_dropconnect @ self.hierarchy_train + bias[..., None]), self.dropout)

            self.session = tf.Session(graph=self.graph)

    def predict(self, stimulus):
        """Stimulus evaluation"""
        with self.graph.as_default():
            return self.session.run(self.hierarchy, feed_dict={self.stimulus: stimulus})
