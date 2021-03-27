import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.generators import _add_nodes_with_bipartite_label
import pandas as pd
import numpy as np

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
G.add_edge(1, 6)
G.add_edge(2, 7)
G.add_edge(3, 8)
G.add_edge(4, 9)
G.add_edge(5, 10)

G = nx.complete_bipartite_graph(5, 5)
nodes = [1, 2, 3, 4, 5]
pos = nx.bipartite_layout(G, nodes)
nx.draw(G, pos, with_labels=True)
plt.show()
