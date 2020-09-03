import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


np.random.seed(42)
random.seed(42)


def watts_strogatz_graph(n, k, p):
    """Return a Watts–Strogatz small-world graph.
    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring topology.
    p : float
        The probability of rewiring each edge
    """
  
    if k >= n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    G = nx.Graph()
    # adding nodes: make categories here
    nodes = list(range(n))
    
    # connect each node to k/2 neighbors
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))

    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if random.random() < p:
                w = random.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = random.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G


def show_plain_graph(G, N, k, p):
    
    plt.title("N = {}, k = {}, p = {}".format(N, k, p))
    nx.draw(G)    
    plt.show()


if __name__ == '__main__':

    N, k, p = 200, 4, 0.1
    net = watts_strogatz_graph(N, k, p)
    show_plain_graph(net, N, k, p)