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
import two_stage_clustering as tsc


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
	mapping = {"b1": 1, "LRC": 2, "TSC": 3}
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
		elif cluster_type == mapping["TSC"]:
			clusters = tsc.clustering(G)
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


def difference_in_mean_estimator(Z, sigma, outcome):
	x1 = Z == 1
	x0 = Z == 0
	y1 = sigma >= 0.8
	y0 = sigma <= 0.2
	m0 = np.logical_and(x0, y0)
	m1 = np.logical_and(x1, y1)
	a = np.mean(outcome[m1])
	b = np.mean(outcome[m0])
	return a - b


def linear_model_estimateor(Z, sigma, outcome):
	assert type(Z) == np.ndarray and Z.ndim == 1
	assert type(sigma) == np.ndarray and sigma.ndim == 1
	assert type(outcome) == np.ndarray and outcome.ndim == 1
	lr = linear_model.LinearRegression()
	X = np.stack((Z, sigma), axis=-1)
	reg = lr.fit(X, outcome)
	beta = reg.coef_[0]
	gamma = reg.coef_[1]
	return beta + gamma


def estimate(G, model, method):
	assert G.__class__.__name__ == "DiGraph", "Graph isn't digraph"
	assert method in ["uniform", "b1", "TSC", "LRC"], "Method provided (%s) not exists" % method

	#G, adjmat = _convert(G, method)
	adjmat = nx.adjacency_matrix(G)
	Z = _sampling(G, model, method)
	outcome = util.outcome_generator(G, adjmat, Z)
	true_ate = _get_true_ate(G, adjmat)
	if method == "uniform":
		estimated_ate = np.mean(outcome[Z==1]) - np.mean(outcome[Z==0])
	elif method == "b1":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = linear_model_estimateor(Z, sigma, outcome)
	elif method == "TSC":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = linear_model_estimateor(Z, sigma, outcome)
	elif method == "LRC":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = difference_in_mean_estimator(Z, sigma, outcome)
	else:
		raise Exception("Model specified not exists")
	return true_ate, estimated_ate
