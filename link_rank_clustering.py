#	Copyright (C) 2004-2016 by
#	Aric Hagberg <hagberg@lanl.gov>
#	Dan Schult <dschult@colgate.edu>
#	Pieter Swart <swart@lanl.gov>
#	All rights reserved.
#	BSD license.
#	NetworkX:http://networkx.github.io/

#	Modified work Copyright 2017 Kenn Chen

import networkx as nx
import numpy as np
import scipy
import math
import random
from collections import defaultdict
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for

from balanced_partition import BalancedPartition

__all__ = ['pagerank', 'pagerank_numpy', 'pagerank_scipy', 'google_matrix']


# def _major_label(G, cluster, labels):
# 	weights = defaultdict(float)
# 	for node in cluster:
# 		cur_label = labels[node]
# 		for followee in G.successors(node):
# 			label = labels[followee]
# 			if label == cur_label: continue
# 			weights[label] += G[node][followee]['weight']
# 		for follower in G.predecessors(node):
# 			label = labels[follower]
# 			if label == cur_label: continue
# 			weights[label] += G[follower][node]['weight']
# 	return max(weights, key=lambda k: weights[k])
#
#
# def clustering(G, M):
# 	N = G.number_of_nodes()
# 	K = math.ceil(N / M)
# 	labels = [-1] * N
# 	counter = 0
# 	clusters = defaultdict(set)
# 	edges = ((u, v, data['weight']) for u, v, data in G.edges(data=True))
# 	edges = sorted(edges, key=lambda x: x[2], reverse=True)
# 	for u, v, w in edges:
# 		lu, lv = labels[u], labels[v]
# 		if lu == -1 and lv == -1:
# 			label = counter
# 			counter += 1
# 			labels[u] = labels[v] = label
# 			clusters[label].update([u, v])
# 		elif lu != -1 and lv != -1:
# 			if lu == lv or len(clusters[lu]) + len(clusters[lv]) > K:
# 				continue
# 			for node in clusters[lv]:
# 				labels[node] = lu
# 			clusters[lu] |= clusters[lv]
# 			del clusters[lv]
# 		elif lu != -1:
# 			if len(clusters[lu]) >= K:
# 				label = counter
# 				counter += 1
# 			else:
# 				label = lu
# 			labels[v] = label
# 			clusters[label].add(v)
# 		else:
# 			if len(clusters[lv]) >= K:
# 				label = counter
# 				counter += 1
# 			else:
# 				label = lv
# 			labels[u] = label
# 			clusters[label].add(u)
#
# 	deleted_clusters = []
# 	for label, cluster in clusters.items():
# 		if len(cluster) < K/2:
# 			old_label = label
# 			new_label = _major_label(G, cluster, labels)
# 			for node in cluster:
# 				labels[node] = new_label
# 			clusters[new_label] |= cluster
# 			deleted_clusters.append(old_label)
# 	for label in deleted_clusters:
# 		del clusters[label]
#
# 	return clusters.values()


def linkrank(G):
	GM, PR = google_matrix(G), pagerank(G)
	LR = scipy.sparse.diags(PR) * GM
	return nx.from_scipy_sparse_matrix(LR, create_using=nx.DiGraph())


def pagerank(G, alpha=0.85, max_iter=100, tol=1.0e-6):
	# Create a copy in (right) stochastic form
	W = nx.stochastic_graph(G)
	N = W.number_of_nodes()

	# Choose fixed starting vector if not given
	x = np.repeat(1.0 / N, N)
	p = np.repeat(1.0 / N, N)
	dangling_weights = p
	dangling_nodes = [i for i in W if W.out_degree(i) == 0.0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = np.zeros(N)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for i in range(N):
			# this matrix multiply looks odd because it is
			# doing a left multiply x^T=xlast^T*W
			for nbr in W[i]:
				x[nbr] += alpha * xlast[i] * W[i][nbr]['weight']
			x[i] += danglesum * dangling_weights[i] + (1.0 - alpha) * p[i]
		# check convergence, l1 norm
		err = sum(abs(x[i] - xlast[i]) for i in range(N))
		if err < N*tol:
			return x
	raise nx.PowerIterationFailedConvergence(max_iter)


def google_matrix(G, alpha=0.85):
	N = G.number_of_nodes()
	M = nx.adjacency_matrix(G)

	sums = np.asarray(M.sum(axis=1)).reshape(-1)
	dangling_nodes = np.where(sums == 0)[0]
	for node in dangling_nodes:
		r = random.randint(0, N-1)
		M[node, r] = 1
		sums[node] = 1
	divisor = scipy.sparse.diags(1 / sums)
	M = divisor * M
	return M


def clustering(G, M):
	G = linkrank(G)
	BP = BalancedPartition(G, M, weighted=True)
	return BP.clustering()


if __name__ == "__main__":
	graph_file = "data/wiki-Vote.txt"
	G = nx.convert_node_labels_to_integers(nx.read_edgelist(graph_file, create_using=nx.DiGraph()))
	G = linkrank(G)
	BP = BalancedPartition(G, 50, weighted=True)
	x = BP.clustering()
	print(list(x))
