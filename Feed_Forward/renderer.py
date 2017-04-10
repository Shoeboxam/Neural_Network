import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

graphics = {'node_color': 'w',
            'alpha': .8,
            'node_size': 600
            }


# hierarchy_pos originates from here: http://stackoverflow.com/a/29597209
def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5 ):
    '''If there is a cycle that is reachable from root, then result will not be a hierarchy.

       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
    '''

    def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,
                  pos = None, parent = None, parsed = [] ):
        if(root not in parsed):
            parsed.append(root)
            if pos == None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            neighbors = G.neighbors(root)
            if len(neighbors)!=0:
                dx = width/len(neighbors)
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    pos = h_recur(G,neighbor, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                        parent = root, parsed = parsed)
        return pos

    return h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5)


def hierarchy_render(G):

    pos = hierarchy_pos(G, root=0)
    labels = nx.get_node_attributes(G, 'fun')

    for i in labels:
        labels[i] = str(i) + '~' + str(labels[i])

    # Delta
    nodes_delta = [node[0] for node in nx.get_node_attributes(G, 'fun').items() if node[1].usage == 'delta']
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=nodes_delta, node_shape='^', **graphics)
    nodes.set_edgecolor('w')

    label_delta = {k: v for k, v in labels.items() if k in nodes_delta}
    nx.draw_networkx_labels(G, pos, nodelist=nodes_delta, labels=label_delta)

    # Basis
    nodes_basis = [node[0] for node in nx.get_node_attributes(G, 'fun').items() if node[1].usage == 'basis']
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=nodes_basis, node_shape='o', **graphics)
    nodes.set_edgecolor('w')

    label_basis = {k: v for k, v in labels.items() if k in nodes_basis}
    nx.draw_networkx_labels(G, pos, nodelist=nodes_basis, labels=label_basis)

    # Weight
    nodes_weight = [node[0] for node in nx.get_node_attributes(G, 'fun').items() if node[1].usage == 'weight']
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=nodes_weight, node_shape='s', **graphics)
    nodes.set_edgecolor('w')

    label_weight = {k: v for k, v in labels.items() if k in nodes_weight}
    nx.draw_networkx_labels(G, pos, nodelist=nodes_weight, labels=label_weight)

    # Remaining nodes
    nodes_rem = set(G.nodes()) - set(nodes_delta) - set(nodes_basis) - set(nodes_weight)
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=nodes_rem, node_shape='*', **graphics)
    nodes.set_edgecolor('w')

    label_rem = {k: v for k, v in labels.items() if k in nodes_rem}
    nx.draw_networkx_labels(G, pos, nodelist=nodes_rem, labels=label_rem)

    # Final
    nx.draw_networkx_edges(G, pos, arrows=False)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.axis('off')
    plt.show()
