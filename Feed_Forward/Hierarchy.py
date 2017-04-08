import networkx as nx
import matplotlib.pyplot as plt
from Function import *
graphics = {'node_color': 'w',
            'alpha': .7,
            'node_size': 850
            }

G = nx.DiGraph()
layers = [2, 23, 12, 22, 12]

G.add_node(-1, {'funct': delta_linear, 'source': 1})

for i in range(len(layers)):
    G.add_node(i, {'funct': basis_bent, 'size': layers[i]})
    G.add_edge(i-1, i)

pos = nx.fruchterman_reingold_layout(G)
labels = nx.get_node_attributes(G, 'funct')

# Lower label positions
pos_off = {}
y_off = -.03
for k, v in pos.items():
    pos_off[k] = (v[0], v[1]+y_off)

# Delta
nodes_delta = [node[0] for node in nx.get_node_attributes(G, 'funct').items() if node[1].usage == 'delta']
label_delta = {k: v for k, v in labels.items() if k in nodes_delta}
nx.draw_networkx_nodes(G, pos, nodelist=nodes_delta, node_shape='^', **graphics)
nx.draw_networkx_labels(G, pos_off, nodelist=nodes_delta, labels=label_delta, label_pos=.8)

# Basis
nodes_basis = [node[0] for node in nx.get_node_attributes(G, 'funct').items() if node[1].usage == 'basis']
label_basis = {k: v for k, v in labels.items() if k in nodes_basis}
nx.draw_networkx_nodes(G, pos, nodelist=nodes_basis, node_shape='o', **graphics)
nx.draw_networkx_labels(G, pos, nodelist=nodes_basis, labels=label_basis, label_pos=.8)

nx.draw_networkx_edges(G, pos)

plt.axis('off')
plt.show()
