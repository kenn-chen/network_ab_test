#!/usr/bin/env python

import math
from collections import defaultdict
import networkx as nx
import numpy as np
import pickle
from networkx.utils import groups

import config

def _init_partition(graph, k):
	N = graph.number_of_nodes()
	labels = [0] * N
	step = math.ceil(N / k)
	x = 0
	for i in range(0, N, step):
		end = i + step if i+step < N else N
		labels[i:end] = [x] * (end - i)
		x += 1
	return labels

def _label_propogation_one_round(graph, labels, N):
	total_gain = 0
	for i in range(N):
		max_gain = 0
		swap_node = None
		cmt_i = labels[i]
		linked_cmt = defaultdict(int)
		for nbr in graph.neighbors(i):
			linked_cmt[labels[nbr]] += 1
		inner_links_i = linked_cmt[cmt_i]

		for j in range(N):
			cmt_j = labels[j]
			if cmt_i == cmt_j or linked_cmt[cmt_j] == 0:
				continue
			inner_links_j = sum(1 for nbr in graph.neighbors(j) if labels[nbr] == cmt_j)
			cross_links_i_j = linked_cmt[cmt_j]
			cross_links_j_i = sum(1 for nbr in graph.neighbors(j) if labels[nbr] == cmt_i)
			gain = cross_links_i_j + cross_links_j_i - inner_links_i - inner_links_j
			if gain > max_gain:
				max_gain = gain
				swap_node = j

		if swap_node != None:
			labels[i] = labels[swap_node]
			labels[swap_node] = cmt_i
			total_gain += max_gain
	return total_gain


def _label_shuffle(labels):
	N = len(labels)
	M = int(N * 0.01)
	nodes = np.arange(N)
	np.random.shuffle(nodes)
	node_selected = nodes[:M]
	labels_selected = [labels[i] for i in node_selected]
	np.random.shuffle(labels_selected)
	for i in range(M):
		labels[node_selected[i]] = labels_selected[i]


def _label_propogation(graph, labels):
	N = graph.number_of_nodes()
	gains = []
	zscore = lambda arr: np.std(arr) / np.mean(arr)
	nround = 0
	while len(gains) < 5 or zscore(gains[-5:]) > 0.1:
		print("Shuffling...")
		_label_shuffle(labels)
		gain = _label_propogation_one_round(graph, labels, N)
		gains.append(gain)
		nround += 1
		if len(gains) < 5:
			print("Round %d finished, gain: %d" % (nround, gain))
		else:
			print("Round %d finished, gain: %d, zscore: %.4f" % (nround, gain, zscore(gains[-5:])))
	return labels


def clustering(graph, k):
	print("Starting balanced partition...")
	print("Initializing partition...")
	labels = _init_partition(graph, k)
	labels = _label_propogation(graph, labels)
	labels = {node:label for node,label in enumerate(labels)}
	print("Partitioning finished.")
	return iter(groups(labels).values())


if __name__ == "__main__":
	graph = nx.scale_free_graph(4000)
	graph = graph.to_undirected()
	labels = clustering(graph, 5)

	#print(labels)
