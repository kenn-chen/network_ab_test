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
	N = G.number_of_nodes()
	Z0 = np.zeros(N)
	Z1 = np.ones(N)
	Y0 = util.outcome_generator(G, adjmat, Z0)
	Y1 = util.outcome_generator(G, adjmat, Z1)
	return np.mean(Y1 - Y0)


def _sampling(G, model, method):
	N = G.number_of_nodes()
	if method == "uniform":
		return np.random.binomial(1, 0.5, N)
	mapping = {"b1": 1, "LRC": 2, "TSC": 3, "b2": 4, "LRC-l": 2}
	cluster_type = mapping[method]
	Z = np.empty(N)
	name = config.dynamic["graph_name"]
	cluster_cache_path = util.get_file_path("cluster_cache", name=name, cluster_type=cluster_type)
	if os.path.exists(cluster_cache_path):
		print("Loading clusters from cache...")
		clusters = pickle.load(open(cluster_cache_path, "rb"))
	else:
		if cluster_type == 1:
			BP = BalancedPartition(G)
			clusters = BP.clustering()
		elif cluster_type == 4:
			BP = BalancedPartition(G, ignore_direction=True)
			clusters = BP.clustering()
		elif cluster_type == 2:
			clusters = lrc.clustering(G)
		elif cluster_type == 3:
			clusters = tsc.clustering(G)
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
	filter0 = np.logical_and(Z == 0, sigma <= 0.2)
	filter1 = np.logical_and(Z == 1, sigma >= 0.8)
	y0 = outcome[filter0]
	y1 = outcome[filter1]
	sigma0 = outcome[filter0]
	sigma1 = outcome[filter1]

	sy0 = ' '.join(map(str, y0))
	sy1 = ' '.join(map(str, y1))
	ssigma0 = ' '.join(map(str, sigma0))
	ssigma1 = ' '.join(map(str, sigma1))

	with open("results/outcome0.txt", "a") as fin0,
		 open("results/outcome1.txt", "a") as fin1:
		fin0.write('|'.join(sy0, ssigma0) + '\n')
		fin1.write('|'.join(sy1, ssigma1) + '\n')

	return np.mean(y1) - np.mean(y0)


def linear_model_estimator(Z, sigma, outcome):
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
	assert method in ["uniform", "b1", "b2", "TSC", "LRC", "LRC-l"], "Method provided (%s) not exists" % method

	adjmat = nx.adjacency_matrix(G)
	Z = _sampling(G, model, method)
	outcome = util.outcome_generator(G, adjmat, Z)
	true_ate = _get_true_ate(G, adjmat)
	if method == "uniform":
		estimated_ate = np.mean(outcome[Z==1]) - np.mean(outcome[Z==0])
	elif method == "b1":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = linear_model_estimator(Z, sigma, outcome)
	elif method == "b2":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = linear_model_estimator(Z, sigma, outcome)
	elif method == "TSC":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = linear_model_estimator(Z, sigma, outcome)
	elif method == "LRC":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = difference_in_mean_estimator(Z, sigma, outcome)
	elif method == "LRC-l":
		sigma = util.treated_proportion(adjmat, Z)
		estimated_ate = linear_model_estimator(Z, sigma, outcome)
	else:
		raise Exception("Model specified not exists")
	return true_ate, estimated_ate
