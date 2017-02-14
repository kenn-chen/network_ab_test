#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import sys

def compute(lambda1, lambda2):
	filename = "results/ate-%g-%g.csv" % (lambda1, lambda2)
	results = defaultdict(list)
	with open(filename) as fin:
		fin.readline()
		for line in fin:
			model, method, graph, true_ate, estimated_ate = line.split(",")
			key = (graph, model, method)
			results[key].append((true_ate, estimated_ate))
	for key, value in results.items():
		rmse = np.sqrt(np.mean([(float(x1)-float(x2))**2 for x1,x2 in value]))
		#bias = np.mean([float(x1)-float(x2) for x1,x2 in value])
		#var = np.var([float(x1) for x1,x2 in value])
		graph, model, method = key
		print("%.2f\t%.2f\t%.6f\t%s\t%s\t%s" % (lambda1, lambda2, rmse, graph, model, method))

if __name__ == "__main__":
	lambda1 = [0, 0.25, 0.75, 1]
	lambda2 = [0, 0.1, 0.5, 1]
	for l1 in lambda1:
		for l2 in lambda2:
			compute(l1, l2)
