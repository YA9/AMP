import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms.bipartite.generators import _add_nodes_with_bipartite_label
from networkx.algorithms import bipartite
from networkx.algorithms.shortest_paths import weighted
import pandas as pd
import numpy as np
from nnfs.datasets import sine


def graph():
    # G = nx.barabasi_albert_graph(100, 2)
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)
    G.add_node(5)
    G.add_node(6)
    G.add_node(7)
    G.add_node(8)
    G.add_node(9)
    G.add_node(10)
    G.add_edge(1, 6, weight=5)
    G.add_edge(2, 7)
    G.add_edge(3, 8)
    G.add_edge(4, 9)
    G.add_edge(5, 10)

    # G = nx.complete_bipartite_graph(5, 5)
    nodes = [1, 2, 3, 4, 5]
    pos = nx.bipartite_layout(G, nodes)
    nx.draw(G, pos, with_labels=True)
    # nx.draw(G)

    # G.add_nodes_from([1, 2, 3, 4], bipartite=0)
    # G.add_nodes_from(["a", "b", "c"], bipartite=1)
    # G.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])
    # bottom_nodes, top_nodes = bipartite.sets(G)
    # top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
    # bottom_nodes = set(G) - top_nodes
    # G = bipartite.projected_graph(G, top_nodes)

    # nx.draw(G, with_labels=True)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    # print((sine.create_data()))


def graph1():
    x, y = sine.create_data()
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    plt.scatter(x, y)
    plt.show()
    # print(len(x))


graph1()
