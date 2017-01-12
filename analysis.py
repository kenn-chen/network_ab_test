#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import sys

def stats(filename):
	results = defaultdict(list)
	with open(filename) as fin:
		fin.readline()
		for line in fin:
			model, true_ate, estimated_ate = line.split(",")
			results[model].append((true_ate, estimated_ate))
	for model, value in results.items():
		rmse = np.sqrt(np.mean([(float(x1)-float(x2))**2 for x1,x2 in value]))
		var = np.var([float(x1) for x1,x2 in value])
		print("%s: rmse->%.6f\tvar->%.6f" % (model, rmse, var))

if __name__ == "__main__":
	filename = sys.argv[1]
	stats(filename)
