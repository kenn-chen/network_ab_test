import networkx as nx
import matplotlib.pyplot as plt
graph = nx.read_edgelist("wiki-Talk.txt")
nx.draw(graph)
plt.savefig("plot.png")
