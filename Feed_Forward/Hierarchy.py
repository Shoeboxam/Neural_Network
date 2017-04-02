# Experimental, thinking about changing the net to take the derivative over a graph.
# Not yet implemented, not used anywhere else in the code.

import networkx as nx

from Feed_Forward.Function import *

hierarchy = nx.DiGraph()
layers = [2, 23, 12]

# Root, add within net
# hierarchy.add_node(0, {"size": 1, "func": delta_linear})

node = 1
for i in range(len(layers)):
    hierarchy.add_node(node, {"size": layers[i], "func": basis_arctan})
    hierarchy.add_edge(node-1, node)
