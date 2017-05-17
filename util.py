import networkx as nx
import pickle
import os.path
import numpy as np

import config

def get_file_path(filetype, **kargs):
	if filetype == "graph_file":
		assert "name" in kargs, "Argument error"
		return "data/" + kargs["name"] + ".txt"
	elif filetype == "cluster_cache":
		assert "cluster_type" in kargs and "name" in kargs, "Argument error"
		return "caches/" + kargs["name"] + "_" + str(kargs["cluster_type"]) + "_cluster.pickle"
	elif filetype == "graph_cache":
		assert "name" in kargs, "Argument error"
		return "caches/" + kargs["name"] + "_graph.pickle"
	else:
		raise Exception("file type error")


def save_graph(G, name):
	graph_cache_path = get_file_path("graph_cache", name=name)
	pickle.dump(G, open(graph_cache_path, "wb" ))


def save_cluster(clusters, name, cluster_type):
	cluster_cache_path = get_file_path("cluster_cache", name=name, cluster_type=cluster_type)
	pickle.dump(clusters, open(cluster_cache_path, "wb" ))


def load_graph(name):
	graph_cache_path = get_file_path("graph_cache", name=name)
	graph_cache_exists = os.path.exists(graph_cache_path)
	if graph_cache_exists:
		print("Loading graph from cache...")
		G = pickle.load(open(graph_cache_path, "rb"))
	elif name == "growing_network":
		print("Generating growing network graph...")
		G = nx.gn_graph(config.graph['node_size'], create_using=nx.DiGraph())
	else:
		graph_file_path = get_file_path("graph_file", name=name)
		assert os.path.exists(graph_file_path), "Graph specified not exists"
		print("Loading graph from file: %s..." % graph_file_path)
		G = nx.convert_node_labels_to_integers(nx.read_edgelist(graph_file_path, create_using=nx.DiGraph()))

	if not graph_cache_exists:
		print("Saving graph...")
		save_graph(G, name)
	return G


def treated_proportion(adjmat, Z):
	assert type(Z) == np.ndarray and Z.ndim == 1, "Z is not 1d array"
	outdegrees = np.asarray(adjmat.sum(axis=1)).reshape(-1)
	outdegrees[outdegrees == 0] = 1
	return (Z * adjmat.T) / outdegrees


def outcome_generator(G, adjmat, Z):
	assert type(Z) == np.ndarray and Z.ndim == 1, "Z is not 1d array"
	degrees = [d for _,d in G.out_degree()]
	N = adjmat.shape[0]
	lambda0 = config.parameter['lambda0']
	lambda1 = config.parameter['lambda1']
	lambda2 = config.parameter['lambda2']
	D = np.array(degrees)
	D[D == 0] = 1
	Y = np.zeros(N)
	def outcome_model(adjmat, Z, Y):
		tmp = Y * adjmat.T #1d array * sparse matrix produces dot product
		Y = lambda0 + lambda1*Z + lambda2*tmp/D + np.random.normal(0, 1, N)
		Y[Y >  0] = 1
		Y[Y <= 0] = 0
		return Y
	for _ in range(config.parameter["iter_round"]):
		Y = outcome_model(adjmat, Z, Y)
	assert Y.ndim == 1, "outcome is not 1d array"
	return Y
