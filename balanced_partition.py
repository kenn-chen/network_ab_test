#!/usr/bin/env python

import math
from collections import defaultdict
from collections import Counter
import networkx as nx
import numpy as np
import pickle
import itertools
from networkx.utils import groups

import config
import util

class BalancedPartition():
	def __init__(self, G, weighted=False, ignore_direction=False):
		N = G.number_of_nodes()
		self.G = G

		if ignore_direction:
			self.to_undirected()
		if not weighted:
			self.to_weighted()

		self.labels, self.clusters = self.init_partition()
		self.connections = self.get_connections()


	def to_undirected(self):
		self.G = self.G.to_undirected().to_directed()

	def to_weighted(self):
		G = self.G = self.G.copy()
		E = G.number_of_edges()
		N = G.number_of_nodes()
		weight = 1 / E
		for u in range(N):
			for v in G[u]:
				G[u][v]['weight'] = weight


	def init_partition(self):
		N = self.G.number_of_nodes()
		M = int(N ** 0.5)
		labels = np.zeros(N, dtype=int)
		clusters = [set() for _ in range(M)]
		step = N // M
		label = 0
		for i in range(0, N, step):
			end = i + step
			labels[i:end] = label
			label += 1
		label = 0
		for i in range(step*M, N):
			labels[i] = label
			label += 1
		assert len(Counter(labels).keys()) == M, "Partition initialization error"
		for u, label in enumerate(labels):
			clusters[label].add(u)
		return labels, clusters

	def get_weight(self, u, v):
		G = self.G
		if not G.has_edge(u, v):
			return 0
		return G[u][v]['weight']

	def get_connections(self):
		G, labels = self.G, self.labels
		N = G.number_of_nodes()
		connections = defaultdict(lambda: defaultdict(float))
		for node in range(N):
			for followee in G.successors(node):
				connections[node][labels[followee]] += G[node][followee]['weight']
			for follower in G.predecessors(node):
				connections[node][labels[follower]] += G[follower][node]['weight']
		return connections


	def change_label(self, u, label):
		G, clusters, labels, connections = self.G, self.clusters, self.labels, self.connections
		lu, lv = labels[u], label
		for followee in G.successors(u):
			connections[followee][lu] -= G[u][followee]['weight']
			connections[followee][lv] += G[u][followee]['weight']
		for follower in G.predecessors(u):
			connections[follower][lu] -= G[follower][u]['weight']
			connections[follower][lv] += G[follower][u]['weight']
		clusters[lu].remove(u)
		clusters[lv].add(u)
		labels[u] = lv


	def swap_label(self, u, v):
		lu, lv = self.labels[u], self.labels[v]
		self.change_label(u, lv)
		self.change_label(v, lu)


	def label_propogate(self):
		G, clusters, labels, connections = self.G, self.clusters, self.labels, self.connections
		N = G.number_of_nodes()
		for u in range(N):
			max_gain = 0
			swap_node = None
			lu = labels[u]
			tension = connections[u][lu]
			potential_swap_labels = [label for label in connections[u] if connections[u][label] > tension]
			for label in potential_swap_labels:
				for v in clusters[label]:
					lv = labels[v]
					uv_cut = self.get_weight(u, v) + self.get_weight(v, u)
					cuts = connections[u][lv] + connections[v][lu] - uv_cut
					swapped_cuts = connections[u][lu] + connections[v][lv] + uv_cut
					gain = cuts - swapped_cuts
					if gain > max_gain:
						max_gain, swap_node = gain, v
			if swap_node:
				self.swap_label(u, swap_node)
		tension = sum(connections[i][labels[i]] for i in range(N)) / 2
		cuts = 1 - tension
		return cuts


	def label_shuffle(self):
		G, clusters, labels, connections = self.G, self.clusters, self.labels, self.connections
		N = len(labels)
		M = int(N * config.parameter["shuffle"])
		nodes = np.arange(N)
		np.random.shuffle(nodes)
		nodes_selected = nodes[:M]
		labels_shuffled = [labels[i] for i in nodes_selected]
		np.random.shuffle(labels_shuffled)
		for i in range(M):
			node = nodes_selected[i]
			new_label = labels_shuffled[i]
			self.change_label(node, new_label)


	def iterate(self, max_iter=15):
		G, clusters, labels, connections = self.G, self.clusters, self.labels, self.connections
		zscore = lambda arr: np.std(arr) / np.mean(arr)
		N = G.number_of_nodes()
		cuts_lst = []
		ref = config.parameter["convergence_reference"]
		threshold = config.parameter["convergence_threshold"]
		for i in range(1, max_iter+1):
			if len(cuts_lst) >= ref and zscore(cuts_lst[-ref:]) < threshold:
				break
			print("Shuffling...")
			self.label_shuffle()
			print("Round %d: label propagation..." % i)
			cuts = self.label_propogate()
			cuts_lst.append(cuts)
			if len(cuts_lst) < ref:
				print("Round %d finished, cuts: %f" % (i, cuts))
			else:
				print("Round %d finished, cuts: %f, zscore: %.4f" % (i, cuts, zscore(cuts_lst[-ref:])))
		print("Partitioning finished.")


	def clustering(self, return_label=False):
		self.iterate()
		if return_label:
			return iter(self.labels)
		else:
			return iter(self.clusters)

if __name__ == "__main__":
	f = "data/soc-Slashdot0811.txt"
	G = nx.convert_node_labels_to_integers(nx.read_edgelist(f, create_using=nx.DiGraph()))
	BP = BalancedPartition(G)
	BP.clustering()
