from collections import defaultdict
import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def prob(graph, node):
	x = 1/graph.out_degree(node)
	return x

def get_weights(graph):
	N = graph.number_of_nodes()
	weights = [0] * N
	for node in range(N):
		weights[node] = 1 + sum(prob(graph, follower) for follower in graph.predecessors(node))
	weight_rank = [node for node, weight in sorted(enumerate(weights), key=lambda x: x[1], reverse=True)]
	return weights, weight_rank

def joint_followees(graph, followees, node, weights):
	if not followees:
		followees = defaultdict(lambda: [0, 0])
	else:
		del followees[node]

	for followee in graph.successors(node):
		followees[followee][0] += 1
		followees[followee][1] = weights[followee]
	return followees

def remove_duplicate_weights(graph, node, kernel, total_weights):
	n = graph.out_degree(node)
	dup1 = sum(1/n for followee in graph.successors(node) if followee in kernel)
	dup2 = sum(prob(graph, follower) for follower in graph.predecessors(node) if follower in kernel)
	return total_weights - dup1 - dup2


def majority_cluster(graph, mapping, clusters, node, cluster_size):
	cluster_count = defaultdict(int)
	for followee in graph.successors(node):
		if followee in mapping:
			cluster_id = mapping[followee]
			if len(clusters[cluster_id]) < 2 * cluster_size:
				cluster_count[cluster_id] += 1
	if not cluster_count:
		return random.randint(0, len(clusters)-1)
	else:
		return max(cluster_count, key=cluster_count.get)



def majority_vote(graph, kernels):
	mapping = {}
	for i in range(len(kernels)):
		for node in kernels[i]:
			mapping[node] = i
	N = graph.number_of_nodes()
	cluster_size = N / len(kernels)
	clusters = [list(kernel) for kernel in kernels]
	for node in range(N):
		if node in mapping:
			continue
		cluster_id = majority_cluster(graph, mapping, clusters, node, cluster_size)
		mapping[node] = cluster_id
		clusters[cluster_id].append(node)
	return clusters

def get_graph(graph_file):
	graph = nx.convert_node_labels_to_integers(nx.read_edgelist(graph_file, create_using=nx.DiGraph()))
	return graph

def get_kernels(graph):
	N = graph.number_of_nodes()
	weights, weight_rank = get_weights(graph)

	iter_round = 100
	cluster_size = N // iter_round
	seed_node_range = N // 10
	kernels = []
	selected_nodes = set()
	for _ in range(iter_round):
		kernel = set()
		seed_node = weight_rank[random.randint(0, N-1)]
		if seed_node in selected_nodes:
			continue
		selected_nodes.add(seed_node)
		kernel.add(seed_node)
		followees = joint_followees(graph, None, seed_node, weights)

		total_weights = weights[seed_node]
		while total_weights < cluster_size:
			found = False
			ranked_followees = [node for node, _ in sorted(followees.items(), key=lambda x: x[1], reverse=True)]
			for followee in ranked_followees:
				if followee not in selected_nodes and followee not in kernel:
					found = True
					kernel.add(followee)
					followees = joint_followees(graph, followees, followee, weights)
					total_weights = remove_duplicate_weights(graph, followee, kernel, total_weights + weights[followee])
					break
			if not found:
				break
		if total_weights >= cluster_size:
			kernels.append(kernel)
			selected_nodes |= kernel
	return kernels

def clustering(graph):
	kernels = get_kernels(graph)
	clusters = majority_vote(graph, kernels)
	return clusters

if __name__ == "__main__":
	graph_file = "data/wiki-Vote.txt"
	graph = get_graph(graph_file)
	kernels = get_kernels(graph)
	clusters = majority_vote(graph, kernels)
	print([len(cluster) for cluster in clusters])
