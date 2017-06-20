#!/usr/bin/env python

import argparse
import os.path
import numpy as np

import config
import util
from estimator import estimate


def write_result(model, method, graph, true_ate, estimated_ate):
	outputfile = config.dynamic['outputfile']
	if not os.path.exists(outputfile):
		with open(outputfile, 'w') as fout:
			fout.write("model,method,graph,true_ate,estimated_ate\n")
			fout.write("%s,%s,%s,%f,%f\n" % (model, method, graph, true_ate, estimated_ate))
	else:
		with open(outputfile, 'a') as fout:
			fout.write("%s,%s,%s,%f,%f\n" % (model, method, graph, true_ate, estimated_ate))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Causal inference estimation.')
	parser.add_argument('-m', '--model', metavar='estimation model', dest="model", default="linear1")
	parser.add_argument('-g', '--graph', metavar='name', dest="name", default="wiki-Vote")
	parser.add_argument('-o', '--output', metavar='result file', dest="outputfile")
	parser.add_argument('-M', '--method', metavar='method', dest="method", default="b1")
	parser.add_argument('-l1', '--lambda1', metavar='lambda1', dest="lambda1", required=True)
	parser.add_argument('-l2', '--lambda2', metavar='lambda2', dest="lambda2", required=True)

	args = parser.parse_args()
	name = args.name
	config.dynamic["graph_name"] = name
	config.parameter["lambda1"] = float(args.lambda1)
	config.parameter["lambda2"] = float(args.lambda2)
	if args.outputfile is not None:
		config.dynamic["outputfile"] = args.outputfile

	config.dynamic["outputfile"] = "results/ate-%s-%s.csv" % (args.lambda1, args.lambda2)

	G = util.load_graph(name)
	assert G.__class__.__name__ == "DiGraph", "Graph isn't digraph"

	print("Starting estimating...")
	true_ate, estimated_ate = estimate(G, args.model, args.method)
	print("Writing result...")
	write_result(args.model, args.method, name, true_ate, estimated_ate)
	print("Done.")
