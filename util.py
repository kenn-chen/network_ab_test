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


def transform(graph, adjmat, directed):
	if graph.is_directed() and not directed:
		graph = graph.to_undirected()
		adjmat = nx.adjacency_matrix(graph)
	elif graph.is_undirected() and directed:
		graph = graph.to_directed()
		adjmat = nx.adjacency_matrix(graph)
	return graph, adjmat


def treated_proportion(Z, adjmat):
	assert type(Z) == np.ndarray and Z.ndim == 1, "Z is not 1d array"
	return np.array(Z.dot(adjmat.T)).reshape(-1)


def outcome_generator(graph, Z, adjmat, is_directed=True):
	assert type(Z) == np.ndarray and Z.ndim == 1, "Z is not 1d array"
	graph, adjmat = transform(graph, adjmat, is_directed)
	if graph.is_undirected():
		degrees = [d for _,d in graph.degree()]
	else:
		degrees = [d for _,d in graph.out_degree()]
	N = adjmat.shape[0]
	lambda0 = config.parameter['lambda0']
	lambda1 = config.parameter['lambda1']
	lambda2 = config.parameter['lambda2']
	D = np.array(degrees)
	D[D==0] = 1
	Y = np.zeros(N)
	def outcome_model(Z, adjmat, Y):
		tmp = Y.dot(adjmat.T).reshape(-1)
		Y = lambda0 + lambda1*Z + lambda2*tmp/D + np.random.normal(0, 1, N)
		Y[Y > 0] = 1
		return Y
	for _ in range(config.parameter["iter_round"]):
		Y = outcome_model(Z, adjmat, Y)
	assert Y.ndim == 1, "outcome is not 1d array"
	return Y
