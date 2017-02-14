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
	parser.add_argument('-g', '--graph', metavar='graph name', dest="graph_name", default="growing_network")
	parser.add_argument('-o', '--output', metavar='result file', dest="outputfile")
	parser.add_argument('-M', '--method', metavar='method', dest="method", default="baseline1")
	parser.add_argument('-l1', '--lambda1', metavar='lambda1', dest="lambda1", required=True)
	parser.add_argument('-l2', '--lambda2', metavar='lambda2', dest="lambda2", required=True)
	parser.add_argument('-b', '--binary', dest="binary", action='store_true')


	args = parser.parse_args()
	graph_name = args.graph_name
	config.dynamic["binary"] = args.binary
	config.dynamic["graph_name"] = graph_name
	config.parameter["lambda1"] = float(args.lambda1)
	config.parameter["lambda2"] = float(args.lambda2)
	if args.outputfile is not None:
		config.dynamic["outputfile"] = args.outputfile
	elif args.binary:
		config.dynamic["outputfile"] = "results/b_ate-%s-%s.csv" % (args.lambda1, args.lambda2)
	else:
		config.dynamic["outputfile"] = "results/ate-%s-%s.csv" % (args.lambda1, args.lambda2)

	graph, adjmat = util.load_graph(graph_name)
	assert graph.__class__.__name__ == "DiGraph", "Graph isn't digraph"

	print("Starting estimating...")
	true_ate, estimated_ate = estimate(graph, adjmat, args.model, args.method)
	print("Writing result...")
	write_result(args.model, args.method, graph_name, true_ate, estimated_ate)
	print("Done.")
