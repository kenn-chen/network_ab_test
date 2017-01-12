import networkx as nx
import pickle
import os.path
import numpy as np

import config

def save_graph(graph):
	pickle.dump(graph, open(config.dynamic['graph_file'], "wb" ))


def save_community(communities):
	pickle.dump(communities, open(config.dynamic['community_file'], "wb" ))


def load_graph(graph_type="barabasi_albert", path=None, directed=False, only_bidirection=False):
	save_flag = True
	if os.path.exists(config.dynamic["graph_file"]):
		print("Loading graph from cache...")
		save_flag = False
		graph = pickle.load(open(config.dynamic["graph_file"], "rb"))
	elif path:
		print("Loading graph from file: %s..." % path)
		if directed:
			graph = nx.convert_node_labels_to_integers(nx.read_edgelist(path, create_using=nx.DiGraph()))
		else:
			graph = nx.convert_node_labels_to_integers(nx.read_edgelist(path))
	else:
		if graph_type == "barabasi_albert":
			print("Generating barabasi_albert graph...")
			graph = nx.barabasi_albert_graph(config.graph['node_size'], 2)
		elif graph_type == "scale_free":
			print("Generating scale_free graph...")
			graph = nx.scale_free_graph(config.graph['node_size'])
			if not directed:
				print("Converting to undirected...")
				graph = graph.to_undirected()
	if not graph:
		raise "Graph loading error"
	adjmat = nx.adjacency_matrix(graph)
	if only_bidirection:
		print("Deleting unidirectional edges...")
		adjmat = adjmat + adjmat.T
		adjmat = np.multiply(adjmat, adjmat==2) / 2
		adjmat = adjmat.astype(int)
		graph = nx.from_scipy_sparse_matrix(adjmat, create_using=nx.Graph())
	if save_flag == True:
		print("Saving graph...")
		save_graph(graph)
	return graph, adjmat


def number_of_neighbors(graph, node):
	return sum(1 for _ in graph.neighbors(node))


def treated_proportion(Z, adjmat):
	return np.array(np.matrix(Z) * adjmat)


def outcome_generator(graph, Z, adjmat):
	N = adjmat.shape[0]
	lambda0 = np.array([0.1] * N)
	lambda1 = 0.5
	lambda2 = 0.2
	D = np.array([number_of_neighbors(graph, i) for i in range(N)])
	Y = np.matrix([0] * N)
	def outcome_model(Z, adjmat, Y):
		return lambda0 + lambda1*Z + lambda2*np.array(Y*adjmat)/D + np.random.normal(0, 1, N)
	for i in range(10):
		Y = outcome_model(Z, adjmat, Y)
	return Y.reshape(-1)
