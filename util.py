import networkx as nx
import pickle
import os.path
import numpy as np

import config

traits = None

def get_file_path(filetype, **kargs):
	if filetype == "graph_file":
		assert "graph_name" in kargs, "Argument error"
		return "data/" + kargs["graph_name"] + ".txt"
	elif filetype == "community_cache":
		assert "community_type" in kargs and "graph_name" in kargs, "Argument error"
		return "caches/" + kargs["graph_name"] + "_" + str(kargs["community_type"]) + "_community.pickle"
	elif filetype == "graph_cache":
		assert "graph_name" in kargs, "Argument error"
		return "caches/" + kargs["graph_name"] + "_graph.pickle"
	else:
		raise Exception("file type error")

def save_graph(graph, graph_name):
	graph_cache_path = get_file_path("graph_cache", graph_name=graph_name)
	pickle.dump(graph, open(graph_cache_path, "wb" ))


def save_community(communities, graph_name, community_type):
	community_cache_path = get_file_path("community_cache", graph_name=graph_name, community_type=community_type)
	pickle.dump(communities, open(community_cache_path, "wb" ))


def load_graph(graph_name):
	graph_cache_path = get_file_path("graph_cache", graph_name=graph_name)
	graph_cache_exists = os.path.exists(graph_cache_path)
	if graph_cache_exists:
		print("Loading graph from cache...")
		graph = pickle.load(open(graph_cache_path, "rb"))
	elif graph_name == "growing_network":
		print("Generating gn_graph graph...")
		graph = nx.gn_graph(config.graph['node_size'], create_using=nx.DiGraph())
#	elif graph_name == "scale_free":
#		print("Generating scale_free graph...")
#		graph = nx.scale_free_graph(config.graph['node_size'], create_using=nx.DiGraph())
	else:
		graph_file_path = get_file_path("graph_file", graph_name=graph_name)
		assert os.path.exists(graph_file_path), "Graph specified not exists"
		print("Loading graph from file: %s..." % graph_file_path)
		graph = nx.convert_node_labels_to_integers(nx.read_edgelist(graph_file_path, create_using=nx.DiGraph()))
	adjmat = nx.adjacency_matrix(graph)
	if not graph_cache_exists:
		print("Saving graph...")
		save_graph(graph, graph_name)
	return graph, adjmat


def treated_proportion(adjmat, Z):
	assert type(Z) == np.ndarray and Z.ndim == 1, "Z is not 1d array"
	return Z * adjmat.T


def outcome_generator(graph, adjmat, Z):
	assert graph.__class__.__name__ == "DiGraph", "Graph isn't digraph"
	assert type(Z) == np.ndarray and Z.ndim == 1, "Z is not 1d array"
	degrees = [d for _,d in graph.out_degree()]
	N = adjmat.shape[0]
	lambda0 = config.parameter['lambda0']
	lambda1 = config.parameter['lambda1']
	lambda2 = config.parameter['lambda2']
	D = np.array(degrees)
	D[D==0] = 1
	Y = np.zeros(N)
	if traits is None:
		traits = np.random.normal(0, 1, N)
	def outcome_model(adjmat, Z, Y):
		tmp = Y * adjmat.T #1d array * sparse matrix produces dot product
		Y = lambda0 + lambda1*Z + lambda2*tmp/D + traits
		return Y
	for _ in range(config.parameter["iter_round"]):
		Y = outcome_model(adjmat, Z, Y)
	assert Y.ndim == 1, "outcome is not 1d array"
	return Y
