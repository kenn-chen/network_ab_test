#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import sys
import os.path

def write_result(graph, model, method, lambda1, lambda2, RMSE, bias, variance):
	outputfile = "results/%s.csv" % graph
	if not os.path.exists(outputfile):
		with open(outputfile, 'w') as fout:
			fout.write("model,method,lambda1,lambda2,RMSE,bias,variance\n")
			fout.write("%s,%s,%.2f,%.2f,%f,%f,%f\n" % (model, method, lambda1, lambda2, RMSE, bias, variance))
	else:
		with open(outputfile, 'a') as fout:
			fout.write("%s,%s,%.2f,%.2f,%f,%f,%f\n" % (model, method, lambda1, lambda2, RMSE, bias, variance))

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
		RMSE = np.sqrt(np.mean([(float(x1)-float(x2))**2 for x1,x2 in value]))
		bias = np.mean([float(x1)-float(x2) for x1,x2 in value])
		variance = np.var([float(x1) for x1,x2 in value])
		graph, model, method = key
		write_result(graph, model, method, lambda1, lambda2, RMSE, bias, variance)

if __name__ == "__main__":
	lambda1 = [0, 0.25, 0.75, 1]
	lambda2 = [0, 0.1, 0.5, 1]
	for l1 in lambda1:
		for l2 in lambda2:
			compute(l1, l2)
