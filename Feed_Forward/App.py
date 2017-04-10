import networkx as nx
from Function import *
from renderer import hierarchy_render

from Neural import Neural

G = nx.DiGraph()
layers = [2, 23, 12, 22, 12]

G.add_node(0, fun=delta_linear)
G.add_node(1, fun=basis_bent)

for i in range(1, len(layers)):
    G.add_node(2 * i - 1, {'fun': basis_arctan, 'size': layers[i]})
    G.add_edge(2 * i - 2, 2 * i - 1)
    G.add_node(2 * i, {'fun': weight_propagate})
    G.add_edge(2 * i - 1, 2 * i)

G.add_node(100, {'fun': source, 'input': None})
G.add_edge(8, 100)

for i in range(1, len(layers)):
    G.add_node(i + 50, {'fun': weight_terminal, 'source': i})
    G.add_edge(2 * i, i + 50)

G.add_node(10, {'fun': basis_bent, 'size': 2000})
G.add_edge(2, 10)
G.add_edge(10, 4)
print(G)

hierarchy_render(G)

net = Neural(G)
