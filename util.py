import networkx as nx
import pickle
import os.path
import numpy as np

import config

def save_graph(graph):
	pickle.dump(graph, open(config.dynamic['graph_file'], "wb" ))


def save_community(communities):
	pickle.dump(communities, open(config.dynamic['community_file'], "wb" ))


def load_graph(graph_type="barabasi_albert", path=None):
	graph_file_exists = os.path.exists(config.dynamic["graph_file"])
	if graph_file_exists:
		print("Loading graph from cache...")
		graph = pickle.load(open(config.dynamic["graph_file"], "rb"))
	elif path:
		print("Loading graph from file: %s..." % path)
		graph = nx.convert_node_labels_to_integers(nx.read_edgelist(path, create_using=nx.DiGraph()))
	else:
		if graph_type == "barabasi_albert":
			print("Generating barabasi_albert graph...")
			graph = nx.barabasi_albert_graph(config.graph['node_size'], 2)
		elif graph_type == "scale_free":
			print("Generating scale_free graph...")
			graph = nx.scale_free_graph(config.graph['node_size'])
	adjmat = nx.adjacency_matrix(graph)
	if not graph_file_exists:
		print("Saving graph...")
		save_graph(graph)
	return graph, adjmat


def treated_proportion(Z, adjmat):
	return np.array(np.matrix(Z) * adjmat)


def outcome_generator(graph, Z, adjmat):
	if config.dynamic["undirected"] == True:
		return _outcome_generator_undirected(graph, Z, adjmat)
	N = adjmat.shape[0]
	lambda0 = np.array([0.1] * N)
	lambda1 = 0.5
	lambda2 = 0.5
	D = np.array([graph.out_degree(i) for i in range(N)])
	D += (D == 0).astype(int)
	adjmat_t = adjmat.T
	Y = np.matrix([0] * N)
	def outcome_model(Z, adjmat_t, Y):
		return lambda0 + lambda1*Z + lambda2*np.array(Y*adjmat_t)/D + np.random.normal(0, 1, N)
	for i in range(20):
		Y = outcome_model(Z, adjmat_t, Y)
	return Y.reshape(-1)

def _outcome_generator_undirected(graph, Z, adjmat):
	graph_u = graph.to_undirected()
	adjmat_u = nx.adjacency_matrix(graph_u)
	N = adjmat_u.shape[0]
	lambda0 = np.array([0.1] * N)
	lambda1 = 0.5
	lambda2 = 0.5
	D = np.array([graph_u.degree(i) for i in range(N)])
	D += (D == 0).astype(int)
	Y = np.matrix([0] * N)
	def outcome_model(Z, adjmat, Y):
		return lambda0 + lambda1*Z + lambda2*np.array(Y*adjmat)/D + np.random.normal(0, 1, N)
	for i in range(20):
		Y = outcome_model(Z, adjmat_u, Y)
	return Y.reshape(-1)
