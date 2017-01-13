import networkx as nx
import numpy as np
import os.path
import math
import pickle
from sklearn import linear_model

import config
import util
import balanced_partition as bp

def _remove_unidirectional_edges(adjmat):
	adjmat_r = adjmat + adjmat.T
	adjmat_r = adjmat_r.multiply(adjmat_r==2) / 2
	adjmat_r = adjmat_r.astype(int)
	graph_r = nx.from_scipy_sparse_matrix(adjmat_r, create_using=nx.Graph())
	return graph_r, adjmat_r

def _get_true_ate(graph, adjmat):
	N = adjmat.shape[0]
	Z0 = np.array([0] * N)
	Z1 = np.array([1] * N)
	Y0 = util.outcome_generator(graph, Z0, adjmat)
	Y1 = util.outcome_generator(graph, Z1, adjmat)
	return np.mean(Y1 - Y0)


def _estimate_lm1(Z, sigma, outcome):
	lr = linear_model.LinearRegression()
	Z, sigma, outcome = Z.reshape(-1), sigma.reshape(-1), outcome.reshape(-1)
	X = np.stack((Z, sigma), axis=-1)
	reg = lr.fit(X, outcome)
	beta = reg.coef_[0]
	gamma = reg.coef_[1]
	return beta + gamma


def _estimate_lm2(Z, sigma, outcome):
	lr = linear_model.LinearRegression()
	Z, sigma, outcome = Z.reshape(-1), sigma.reshape(-1), outcome.reshape(-1)
	sigma0, outcome0 = sigma[Z == 0].reshape(-1, 1), outcome[Z == 0].reshape(-1, 1)
	sigma1, outcome1 = sigma[Z == 1].reshape(-1, 1), outcome[Z == 1].reshape(-1, 1)
	reg0 = lr.fit(sigma0, outcome0)
	reg1 = lr.fit(sigma1, outcome1)
	alpha0 = reg0.intercept_
	alpha1 = reg1.intercept_
	gamma1 = reg1.coef_[0]
	return alpha1 + gamma1 - alpha0


def _sampling(graph, model="uniform"):
	if model == "uniform":
		return np.random.binomial(1, 0.5, graph.number_of_nodes())
	elif model == "linear1" or model == "linear2":
		Z = [0] * graph.number_of_nodes()
		if os.path.exists(config.dynamic['community_file']):
			communities = pickle.load(open(config.dynamic['community_file'], "rb"))
		else:
			communities = bp.clustering(graph, config.graph['partition_size'])
			util.save_community(communities)
		for cmt in communities:
			assignment = np.random.binomial(1, 0.5)
			for node in cmt:
				Z[node] = assignment
		return np.array(Z)


def _estimate_baseline1(graph, adjmat, model):
	graph_u = graph.to_undirected()
	adjmat_u = nx.adjacency_matrix(graph_u)
	Z = _sampling(graph_u, model)
	outcome = util.outcome_generator(graph, Z, adjmat)
	true_ate = _get_true_ate(graph, adjmat)
	if model == "uniform":
		estimated_ate = np.mean(outcome[Z==1]) - np.mean(outcome[Z==0])
	elif model == "linear1":
		sigma = util.treated_proportion(Z, adjmat_u)
		estimated_ate = _estimate_lm1(Z, sigma, outcome)
	elif model == "linear2":
		sigma = util.treated_proportion(Z, adjmat_u)
		estimated_ate = _estimate_lm2(Z, sigma, outcome)
	return true_ate, estimated_ate

def _estimate_baseline2(graph, adjmat, model):
	graph_r, adjmat_r = _remove_unidirectional_edges(adjmat)
	Z = _sampling(graph_r, model)
	outcome = util.outcome_generator(graph, Z, adjmat)
	true_ate = _get_true_ate(graph, adjmat)
	if model == "uniform":
		estimated_ate = np.mean(outcome[Z==1]) - np.mean(outcome[Z==0])
	elif model == "linear1":
		sigma = util.treated_proportion(Z, adjmat_r)
		estimated_ate = _estimate_lm1(Z, sigma, outcome)
	elif model == "linear2":
		sigma = util.treated_proportion(Z, adjmat_r)
		estimated_ate = _estimate_lm2(Z, sigma, outcome)
	return true_ate, estimated_ate


def estimate_weighted(graph, adjmat):
	N = graph.number_of_nodes()
	graph_u = graph.to_undirected()
	Z = _sampling(graph, "linear1")
	outcome = util.outcome_generator(graph, Z, adjmat)
	true_ate = _get_true_ate(graph, adjmat)
	adjmat_t = adjmat.T
	D = np.array([graph.out_degree(i) for i in range(N)])
	D += (D == 0).astype(int)
	Z = Z.astype('float64')
	alpha = 0.5
	for i in range(10):
		W = Z + alpha/(1+alpha)/D*(np.array(np.matrix(Z)*adjmat_t).reshape(-1))
	C = 1+alpha-W
	W, C, outcome = np.array(W), np.array(C), np.array(outcome)
	ate = sum(W*outcome)/N - sum(C*outcome)/N
	print(W)
	return true_ate, ate

def estimate(graph, adjmat, model="uniform", method="baseline1"):
	if method == "baseline1":
		return _estimate_baseline1(graph, adjmat, model)
	elif method == "baseline2":
		return _estimate_baseline2(graph, adjmat, model)
	elif method == "weighted":
		return estimate_weighted(graph, adjmat)
