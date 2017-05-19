import networkx as nx
import numpy as np
import os.path
import math
import pickle
from sklearn import linear_model

import config
import util
from balanced_partition import BalancedPartition
import link_rank_clustering as lrc

def _remove_unidirectional_edges(G):
	adjmat = nx.adjacency_matrix(G)
	adjmat = adjmat + adjmat.T
	adjmat[adjmat == 1] = 0
	adjmat[adjmat == 2] = 1
	G = nx.from_scipy_sparse_matrix(adjmat, create_using=nx.Graph())
	return G

def _get_true_ate(G, adjmat):
	N = adjmat.shape[0]
	Z0 = np.zeros(N)
	Z1 = np.ones(N)
	results = []
	for _ in range(10):
		Y0 = util.outcome_generator(G, adjmat, Z0)
		Y1 = util.outcome_generator(G, adjmat, Z1)
		results.append(np.mean(Y1 - Y0))
	return np.mean(results)


def _sampling(G, model, method):
	N = G.number_of_nodes()
	if method == "uniform":
		return np.random.binomial(1, 0.5, N)
	mapping = {"b1": 1, "LRC": 2}
	cluster_type = mapping[method]
	Z = np.empty(N)
	name = config.dynamic["graph_name"]
	cluster_cache_path = util.get_file_path("cluster_cache", name=name, cluster_type=cluster_type)
	if os.path.exists(cluster_cache_path):
		print("Loading clusters from cache...")
		clusters = pickle.load(open(cluster_cache_path, "rb"))
	else:
		if cluster_type == mapping["b1"]:
			BP = BalancedPartition(G)
			clusters = BP.clustering()
		elif cluster_type == mapping["LRC"]:
			clusters = lrc.clustering(G)
		util.save_cluster(clusters, name, cluster_type)
	cnt = 0
	for cluster in clusters:
		assignment = np.random.binomial(1, 0.5)
		cluster = list(cluster)
		Z[cluster] = assignment
		cnt += len(cluster)
	assert cnt == N, "cnt not equals N"
	return Z


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


# def _convert(G, method):
# 	if method == "baseline1":
# 		G = G.to_undirected()
# 	adjmat = nx.adjacency_matrix(G)
# 	return G, adjmat


def estimate(G, model, method):
	assert G.__class__.__name__ == "DiGraph", "Graph isn't digraph"
	assert method in ["uniform", "b1", "LRC"], "Method provided (%s) not exists" % method

	#G, adjmat = _convert(G, method)
	adjmat = nx.adjacency_matrix(G)
	Z = _sampling(G, model, method)
	outcome = util.outcome_generator(G, adjmat, Z)
	true_ate = _get_true_ate(G, adjmat)
	if method == "uniform":
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
