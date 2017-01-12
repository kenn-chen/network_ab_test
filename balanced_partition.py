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

def _get_linked_cmt(graph, labels):
	N = graph.number_of_nodes()
	linked_cmt = defaultdict(lambda: defaultdict(int))
	for i in range(N):
		for nbr in graph.neighbors(i):
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
			for nbr in graph.neighbors(i):
				linked_cmt[nbr][labels[i]] -= 1
				linked_cmt[nbr][labels[swap_node]] += 1
			for nbr in graph.neighbors(swap_node):
				linked_cmt[nbr][labels[swap_node]] -= 1
				linked_cmt[nbr][labels[i]] += 1
			labels[i], labels[swap_node] = labels[swap_node], labels[i]
			total_gain += max_gain
	return total_gain


def _label_shuffle(graph, labels, linked_cmt):
	N = len(labels)
	M = int(N * 0.01)
	nodes = np.arange(N)
	np.random.shuffle(nodes)
	node_selected = nodes[:M]
	labels_shuffled = [labels[i] for i in node_selected]
	np.random.shuffle(labels_shuffled)
	for i in range(M):
		node = node_selected[i]
		cmt = labels[node]
		new_cmt = labels_shuffled[i]
		for nbr in graph.neighbors(node):
			linked_cmt[nbr][cmt] -= 1
			linked_cmt[nbr][new_cmt] += 1
		labels[node] = new_cmt


def _label_propogation(graph, labels):
	zscore = lambda arr: np.std(arr) / np.mean(arr)
	N = graph.number_of_nodes()
	gains = []
	nround = 1
	linked_cmt = _get_linked_cmt(graph, labels)
	while len(gains) < 3 or zscore(gains[-3:]) > 0.15:
		print("Shuffling...")
		_label_shuffle(graph, labels, linked_cmt)
		print("round %d: label propagation..." % nround)
		gain = _label_propogation_one_round(graph, labels, linked_cmt)
		gains.append(gain)
		if len(gains) < 3:
			print("Round %d finished, gain: %d" % (nround, gain))
		else:
			print("Round %d finished, gain: %d, zscore: %.4f" % (nround, gain, zscore(gains[-3:])))
		nround += 1
	return labels


def clustering(graph, k):
	print("Starting balanced partition...")
	print("Initializing partition...")
	labels = _init_partition(graph, k)
	labels = _label_propogation(graph, labels)
	labels = {node:label for node,label in enumerate(labels)}
	print("Partitioning finished.")
	return iter(groups(labels).values())
