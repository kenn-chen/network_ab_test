from collections import defaultdict
import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def get_weights(graph):
	N = graph.number_of_nodes()
	weights = [0] * N
	for node in range(N):
		weights[node] = 1 + sum(1/graph.out_degree(follower) for follower in graph.predecessors(node))
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

def majority_kernel(graph, mapping, node, k):
	kernel_count = defaultdict(int)
	for followee in graph.successors(node):
		if followee in mapping:
			kernel_id = mapping[followee]
			kernel_count[kernel_id] += 1
	if not kernel_count:
		return random.randint(0, k-1)
	else:
		return max(kernel_count, key=kernel_count.get)



def majority_vote(graph, kernels):
	mapping = {}
	for i in range(len(kernels)):
		for node in kernels[i]:
			mapping[node] = i
	k = len(kernels)
	N = graph.number_of_nodes()
	clusters = [list(kernel) for kernel in kernels]
	for node in range(N):
		if node in mapping:
			continue
		kernel_id = majority_kernel(graph, mapping, node, k)
		mapping[node] = kernel_id
		clusters[kernel_id].append(node)
	return clusters

def get_graph(graph_file):
	graph = nx.convert_node_labels_to_integers(nx.read_edgelist(graph_file, create_using=nx.DiGraph()))
	return graph

def get_kernels(graph):
	weights, weight_rank = get_weights(graph)
	N = graph.number_of_nodes()

	iter_round = N // 1000
	seed_node_range = int(N // 10)
	kernels = []
	selected_nodes = set()
	for _ in range(iter_round):
		kernel = set()
		seed_node = weight_rank[random.randint(0, seed_node_range)]
		if seed_node in selected_nodes:
			continue
		selected_nodes.add(seed_node)
		kernel.add(seed_node)
		followees = joint_followees(graph, None, seed_node, weights)

		success = True
		total_weights = weights[seed_node]
		while total_weights < 1000:
			#print(total_weights)
			found = False
			ranked_followees = [node for node, _ in sorted(followees.items(), key=lambda x: x[1], reverse=True)]
			for followee in ranked_followees:
				if followee not in selected_nodes and followee not in kernels:
					found = True
					kernel.add(followee)
					selected_nodes.add(followee)
					followees = joint_followees(graph, followees, followee, weights)
					total_weights += weights[followee]
					break
			if not found:
				success = False
				break
		if success:
			kernels.append(kernel)
	return kernels

def clustering(graph):
	kernels = get_kernels(graph)
	clusters = majority_vote(graph, kernels)
	return clusters

if __name__ == "__main__":
	graph_file = "data/soc-Epinions1.txt"
	graph = get_graph(graph_file)
	kernels = get_kernels(graph)
	clusters = majority_vote(graph, kernels)
	print([len(cluster) for cluster in clusters])
