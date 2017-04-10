import matplotlib.pyplot as plt
import numpy as np

import networkx as nx

import Function

plt.style.use('fivethirtyeight')


class Neural(object):

    def __init__(self, graph: nx.DiGraph()):
        self.graph = graph
        self.root = nx.topological_sort(graph)[0]
        queue = [self.root]

        while queue:
            print(queue[0])
            print(graph[queue[0]])
