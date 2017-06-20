import networkx as nx
from collections import Counter
from networkx.utils import groups

import util
from balanced_partition import BalancedPartition as BP


def clustering(G):
	N = G.number_of_nodes()
	threshold = N * 0.001
	core_nodes = {node for node,degree in G.in_degree() if degree >= threshold}
	print(len(core_nodes))
	core_graph = G.copy()
	for node in G.nodes():
		if node not in core_nodes:
			core_graph.remove_node(node)
	mapping1 = {node:i for i, node in enumerate(core_nodes)}
	mapping2 = {i:node for i, node in enumerate(core_nodes)}
	core_graph = nx.relabel_nodes(core_graph, mapping1)

	bp = BP(core_graph)
	labels = bp.clustering(return_label=True)
	labels = {mapping2[node]:label for node, label in enumerate(labels)}

	for node in G.nodes():
		if node not in core_nodes:
			nbr_labels = [labels[nbr] for nbr in G.successors(node) if nbr in core_nodes] or \
						 [labels[nbr] for nbr in G.successors(node) if nbr in labels] or \
						 [-1]
			label = Counter(nbr_labels).most_common(1)[0][0]
			labels[node] = label
	for node in G.nodes():
		if labels[node] == -1:
			nbr_labels = [labels[nbr] for nbr in G.successors(node) if nbr in labels] or \
						 [-1]
			label = Counter(nbr_labels).most_common(1)[0][0]
			labels[node] = label

	clusters = iter(groups(labels).values())
	return clusters
