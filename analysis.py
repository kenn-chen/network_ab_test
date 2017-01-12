#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import sys

def stats(filename):
	results = defaultdict(list)
	with open(filename) as fin:
		fin.readline()
		for line in fin:
			model, method, graph, true_ate, estimated_ate = line.split(",")
			name = "%s|%s|%s" % (graph, model, method)
			results[name].append((true_ate, estimated_ate))
	for name, value in results.items():
		rmse = np.sqrt(np.mean([(float(x1)-float(x2))**2 for x1,x2 in value]))
		bias = np.mean([abs(float(x1)-float(x2)) for x1,x2 in value])
		var = np.var([float(x1) for x1,x2 in value])
		print("rmse->%.6f\tbias->%.6f\tvar->%.6f: %s" % (rmse, bias, var, name))

if __name__ == "__main__":
	if len(sys.argv) > 1:
		filename = sys.argv[1]
	else:
		filename = "results/ate.csv"
	stats(filename)
