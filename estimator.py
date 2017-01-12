import networkx as nx
import numpy as np
import os.path
import math
import pickle
from sklearn import linear_model

import config
import util
import balanced_partition as bp

def get_true_ate(graph, adjmat):
	N = adjmat.shape[0]
	Z0 = np.array([0] * N)
	Z1 = np.array([1] * N)
	Y0 = util.outcome_generator(graph, Z0, adjmat)
	Y1 = util.outcome_generator(graph, Z1, adjmat)
	return np.mean(Y1 - Y0)


def train_lm1(Z, sigma, outcome):
	lr = linear_model.LinearRegression()
	Z, sigma, outcome = Z.reshape(-1), sigma.reshape(-1), outcome.reshape(-1)
	X = np.stack((Z, sigma), axis=-1)
	reg = lr.fit(X, outcome)
	alpha = reg.intercept_
	beta = reg.coef_[0]
	gamma = reg.coef_[1]
	return alpha, beta, gamma

def sampling(graph, model="uniform"):
	if model == "uniform":
		return np.random.binomial(1, 0.5, graph.number_of_nodes())
	elif model == "linear1":
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


def estimate(graph, adjmat, model="uniform"):
	print("Sampling using model %s..." % model)
	Z = sampling(graph, model)
	print("Generating outcome...")
	outcome = util.outcome_generator(graph, Z, adjmat)
	print("Getting true ATE...")
	true_ate = get_true_ate(graph, adjmat)
	print("Getting estimated ATE using model %s" % model)
	if model == "uniform":
		estimated_ate = np.mean(outcome[Z==1]) - np.mean(outcome[Z==0])
	elif model == "linear1":
		sigma = util.treated_proportion(Z, adjmat)
		alpha, beta, gamma = train_lm1(Z, sigma, outcome)
		estimated_ate = beta + gamma
	return true_ate, estimated_ate
