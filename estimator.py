import networkx as nx
import numpy as np
import os.path
import math
import pickle
from sklearn import linear_model

import config
import util
from balanced_partition import clustering
from new_method import partition

def _remove_unidirectional_edges(graph):
	adjmat = nx.adjacency_matrix(graph)
	adjmat = adjmat + adjmat.T
	adjmat[adjmat == 1] = 0
	adjmat[adjmat == 2] = 1
	graph = nx.from_scipy_sparse_matrix(adjmat, create_using=nx.Graph())
	return graph

def _get_true_ate(graph, adjmat):
	N = adjmat.shape[0]
	Z0 = np.zeros(N)
	Z1 = np.ones(N)
	Y0 = util.outcome_generator(graph, adjmat, Z0)
	Y1 = util.outcome_generator(graph, adjmat, Z1)
	return np.mean(Y1 - Y0)


def _sampling(graph, community_type, model):
	assert graph.__class__.__name__ == "DiGraph", "Graph isn't digraph"
	if model == "uniform":
		return np.random.binomial(1, 0.5, graph.number_of_nodes())
	elif model == "linear1" or model == "linear2":
		if community_type == 1:
			graph = graph.to_undirected()
		elif community_type == 2:
			graph = _remove_unidirectional_edges(graph)
		N = graph.number_of_nodes()
		Z = np.empty(N)
		graph_name = config.dynamic["graph_name"]
		community_cache_path = util.get_file_path("community_cache", graph_name=graph_name, community_type=community_type)
		if os.path.exists(community_cache_path):
			print("Loading communities from cache...")
			communities = pickle.load(open(community_cache_path, "rb"))
		else:
			if community_type == 3:
				communities = clustering(graph, config.graph['partition_size'])
			elif community_type == 4:
				communities = partition(graph)
			util.save_community(communities, graph_name, community_type)
		cnt = 0
		for cmt in communities:
			assignment = np.random.binomial(1, 0.5)
			for node in cmt:
				Z[node] = assignment
				cnt += 1
		assert cnt == N, "cnt not equals N"
		return Z
	else:
		raise Exception("Model specified not exists")


def _estimate_lm1(Z, sigma, outcome):
	assert type(Z) == np.ndarray and Z.ndim == 1
	assert type(sigma) == np.ndarray and sigma.ndim == 1
	assert type(outcome) == np.ndarray and outcome.ndim == 1
	lr = linear_model.LinearRegression()
	X = np.stack((Z, sigma), axis=-1)
	reg = lr.fit(X, outcome)
	beta = reg.coef_[0]
	gamma = reg.coef_[1]
	return beta + gamma


def _estimate_lm2(Z, sigma, outcome):
	assert type(Z) == np.ndarray and Z.ndim == 1
	assert type(sigma) == np.ndarray and sigma.ndim == 1
	assert type(outcome) == np.ndarray and outcome.ndim == 1
	lr = linear_model.LinearRegression()
	sigma0, outcome0 = sigma[Z == 0].reshape(-1, 1), outcome[Z == 0].reshape(-1, 1)
	sigma1, outcome1 = sigma[Z == 1].reshape(-1, 1), outcome[Z == 1].reshape(-1, 1)
	reg0 = lr.fit(sigma0, outcome0)
	reg1 = lr.fit(sigma1, outcome1)
	alpha0 = reg0.intercept_
	alpha1 = reg1.intercept_
	gamma1 = reg1.coef_[0]
	return alpha1 + gamma1 - alpha0


def _estimate_weighted(graph, adjmat):
	N = graph.number_of_nodes()
	graph_u = graph.to_undirected()
	Z = _sampling(graph, "linear1") #todo:...
	outcome = util.outcome_generator(graph, Z, adjmat)
	true_ate = _get_true_ate(graph, adjmat)
	adjmat_t = adjmat.T
	D = np.array([graph.out_degree(i) for i in range(N)])
	D += (D == 0).astype(int)
	Z = Z.astype('float64')
	alpha = 0.5
	W = Z
	for i in range(10):
		W = Z + alpha/(1+alpha)/D*(np.array(np.matrix(W)*adjmat_t).reshape(-1))
	C = 1+alpha-W
	W, C, outcome = np.array(W), np.array(C), np.array(outcome)
	ate = sum(((1+alpha)/W - C/(1+alpha)) * outcome)
	return true_ate, ate

def estimate(graph, adjmat, model, method):
	assert graph.__class__.__name__ == "DiGraph", "Graph isn't digraph"
	assert method in ["baseline1", "baseline2", "baseline3", "new", "weighted"], "Method provided (%s) not exists" % method
	if method == "weighted":
		return _estimate_weighted(graph, adjmat)

	community_type = {"baseline1": 1, "baseline2": 2, "baseline3": 3, "new": 4}[method]
	Z = _sampling(graph, community_type, model)
	outcome = util.outcome_generator(graph, adjmat, Z)
	true_ate = _get_true_ate(graph, adjmat)
	if model == "uniform":
		estimated_ate = np.mean(outcome[Z==1]) - np.mean(outcome[Z==0])
	elif model == "linear1":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = _estimate_lm1(Z, sigma, outcome)
	elif model == "linear2":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = _estimate_lm2(Z, sigma, outcome)
	else:
		raise Exception("Model specified not exists")
	return true_ate, estimated_ate
