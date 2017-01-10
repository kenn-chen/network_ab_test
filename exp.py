import networkx as nx
import numpy as np
from sklearn import linear_model

def number_of_neighbors(graph, node):
	return sum(1 for _ in graph.neighbors(node))


def load_as_undirected_graph(path):
	graph = nx.convert_node_labels_to_integers(nx.read_edgelist(path))
	adjmat = nx.adjacency_matrix(graph)
	return graph, adjmat


def load_as_undirected_graph_only_bidirection(path):
	graph = nx.read_edgelist(path, create_using=nx.DiGraph())
	adjmat = nx.adjacency_matrix(graph)
	adjmat = adjmat + adjmat.T
	adjmat = np.multiply(adjmat, adjmat==2) / 2
	adjmat = adjmat.astype(int)
	graph = nx.from_scipy_sparse_matrix(adjmat, create_using=nx.Graph())
	return graph, adjmat


def assign(graph):
	Z = [0] * graph.number_of_nodes()
	communities = nx.algorithms.community.asyn_lpa_communities(graph)
	count_c = 0
	max_c = 0
	for c in communities:
		count_c += 1
		max_c = max(max_c, len(c))

		asm = np.random.binomial(1, 0.5)
		for node in c:
			Z[node] = asm
	print("count_c:%d, max_c:%d" %(count_c, max_c))
	return np.array(Z)


def outcome_generator(graph, Z, adjmat):
	N = adjmat.shape[0]
	lambda0 = np.array([0.1] * N)
	lambda1 = 0.5
	lambda2 = 0.2
	D = np.array([number_of_neighbors(graph, i) for i in range(N)])
	Y = np.matrix([0] * N)
	def outcome_model(Z, adjmat, Y):
		return lambda0 + lambda1*Z + lambda2*np.array(Y*adjmat)/D + np.random.normal(0, 1, N)
	for i in range(10):
		Y = outcome_model(Z, adjmat, Y)
	return Y.reshape(-1)


def treated_proportion(Z, adjmat):
	return np.array(np.matrix(Z) * adjmat)


def train(Z, Sigma, outcome):
	lr = linear_model.LinearRegression()
	Z, Sigma, outcome = Z.reshape(-1), Sigma.reshape(-1), outcome.reshape(-1)
	X = np.stack((Z, Sigma), axis=-1)
	reg = lr.fit(X, outcome)
	alpha = reg.intercept_
	beta = reg.coef_[0]
	gamma = reg.coef_[1]
	return alpha, beta, gamma


def get_true_ate(graph, adjmat):
	N = adjmat.shape[0]
	Z0, Z1 = np.array([0] * N), np.array([1] * N)
	Y0, Y1 = outcome_generator(graph, Z0, adjmat), outcome_generator(graph, Z1, adjmat)
	ate = np.mean(Y1 - Y0)
	return ate


if __name__ == "__main__":
	graph, adjmat = load_as_undirected_graph("wiki-Vote.txt")
	#graph, adjmat = load_as_undirected_graph_only_bidirection("wiki-Vote.txt")
	Z = assign(graph)
	Sigma = treated_proportion(Z, adjmat)

	outcome = outcome_generator(graph, Z, adjmat)
	alpha, beta, gamma = train(Z, Sigma, outcome)
	estimated_ate = beta + gamma
	true_ate = get_true_ate(graph, adjmat)
	print(true_ate, estimated_ate)
