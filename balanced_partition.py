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

def _neighbors(graph, node):
	if graph.is_directed():
		return itertools.chain(graph.successors(node), graph.predecessors(node))
	else:
		return graph.neighbors(node)


def _init_partition(graph, k):
	N = graph.number_of_nodes()
	labels = np.zeros(N)
	step = N // k
	label = 0
	for i in range(0, N, step):
		end = i + step
		labels[i:end] = label
		label += 1
	label = 0
	for i in range(step*k, N):
		labels[i] = label
		label += 1
	return labels

def _get_linked_cmt(graph, labels):
	N = graph.number_of_nodes()
	linked_cmt = defaultdict(lambda: defaultdict(int))
	for i in range(N):
		for nbr in _neighbors(graph, i):
			linked_cmt[i][labels[nbr]] += 1
	return linked_cmt


def _label_propogation_one_round(graph, labels, linked_cmt):
	N = graph.number_of_nodes()
	total_gain = 0
	for i in range(N):
		max_gain = 0
		swap_node = None
		cmt_i = labels[i]
		inner_links_i = linked_cmt[i][cmt_i]
		potential_cmts = {cmt:True for cmt in linked_cmt[i].keys() if linked_cmt[i][cmt] > inner_links_i}
		if len(potential_cmts) == 0:
			continue
		for j in range(N):
			if labels[j] not in potential_cmts:
				continue
			cmt_j = labels[j]
			cross_links = linked_cmt[i][cmt_j] + linked_cmt[j][cmt_i]
			inner_links = linked_cmt[i][cmt_i] + linked_cmt[j][cmt_j]
			gain = cross_links - inner_links
			if gain > max_gain:
				max_gain = gain
				swap_node = j
		if swap_node != None:
			nbrs = _neighbors(graph, i)
			for nbr in nbrs:
				linked_cmt[nbr][labels[i]] -= 1
				linked_cmt[nbr][labels[swap_node]] += 1
			nbrs = _neighbors(graph, swap_node)
			for nbr in nbrs:
				linked_cmt[nbr][labels[swap_node]] -= 1
				linked_cmt[nbr][labels[i]] += 1
			labels[i], labels[swap_node] = labels[swap_node], labels[i]
			total_gain += max_gain
	return total_gain


def _label_shuffle(graph, labels, linked_cmt):
	N = len(labels)
	M = int(N * config.parameter["shuffle"])
	nodes = np.arange(N)
	np.random.shuffle(nodes)
	nodes_selected = nodes[:M]
	labels_shuffled = [labels[i] for i in nodes_selected]
	np.random.shuffle(labels_shuffled)
	for i in range(M):
		node = nodes_selected[i]
		cmt = labels[node]
		new_cmt = labels_shuffled[i]
		for nbr in _neighbors(graph, i):
			linked_cmt[nbr][cmt] -= 1
			linked_cmt[nbr][new_cmt] += 1
		labels[node] = new_cmt


def _label_propogation(graph, labels):
	zscore = lambda arr: np.std(arr) / np.mean(arr)
	N = graph.number_of_nodes()
	gains = []
	nround = 1
	linked_cmt = _get_linked_cmt(graph, labels)
	ref = config.parameter["convergence_reference"]
	threshold = config.parameter["convergence_threshold"]
	while len(gains) < ref or zscore(gains[-ref:]) > threshold:
		print("Shuffling...")
		_label_shuffle(graph, labels, linked_cmt)
		print("round %d: label propagation..." % nround)
		gain = _label_propogation_one_round(graph, labels, linked_cmt)
		gains.append(gain)
		if len(gains) < ref:
			print("Round %d finished, gain: %d" % (nround, gain))
		else:
			print("Round %d finished, gain: %d, zscore: %.4f" % (nround, gain, zscore(gains[-ref:])))
		nround += 1
	return labels


def clustering(graph, k):
	print("Starting balanced partition...")
	adjmat = nx.adjacency_matrix(graph)
	labels = _init_partition(graph, k)
	assert len(Counter(labels).keys()) == k, "Partition initialization error"
	labels = _label_propogation(graph, labels)
	labels = {node:label for node,label in enumerate(labels)}
	print("Partitioning finished.")
	return iter(groups(labels).values())
