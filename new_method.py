import networkx as nx
from collections import Counter
from networkx.utils import groups

import util
from balanced_partition import clustering

import sys


def partition(graph):
	core_nodes = {node:True for node,degree in graph.in_degree() if degree>=10}
	core_graph = graph.copy()
	for node in graph.nodes():
		if node not in core_nodes:
			core_graph.remove_node(node)
	mapping1 = {node:i for i, node in enumerate(core_nodes)}
	mapping2 = {i:node for i, node in enumerate(core_nodes)}
	core_graph = nx.relabel_nodes(core_graph, mapping1)
	labels = clustering(core_graph, 100, return_label=True)
	labels = {mapping2[node]:label for node, label in labels.items()}

	for node in graph.nodes():
		if node not in core_nodes:
			nbr_labels = [labels[nbr] for nbr in graph.successors(node) if nbr in core_nodes] or \
						 [labels[nbr] for nbr in graph.successors(node) if nbr in labels] or \
						 [-1]
			label = Counter(nbr_labels).most_common(1)[0][0]
			labels[node] = label
	for node in graph.nodes():
		if labels[node] == -1:
			nbr_labels = [labels[nbr] for nbr in graph.successors(node) if nbr in labels] or \
						 [-1]
			label = Counter(nbr_labels).most_common(1)[0][0]
			labels[node] = label

	#communities = iter(groups(labels).values())
	communities = groups(labels).values()
	x = [len(c) for c in sorted(communities, key=len, reverse=True)]
	#return communities
	print(x)
	sys.exit(0)
