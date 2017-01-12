#!/usr/bin/env python

import argparse
import os.path
import numpy as np

import config
import util
from estimator import estimate


def write_result(model, method, true_ate, estimated_ate):
	outputfile = config.dynamic['outputfile']
	if not os.path.exists(outputfile):
		with open(outputfile, 'w') as fout:
			fout.write("model,true_ate,estimated_ate\n")
			fout.write("%s,%s,%f,%f\n" % (model, method, true_ate, estimated_ate))
	else:
		with open(outputfile, 'a') as fout:
			fout.write("%s,%s,%f,%f\n" % (model, method, true_ate, estimated_ate))



if __name__ == "__main__":
	np.seterr(divide='ignore', invalid='ignore')

	parser = argparse.ArgumentParser(description='Causal inference estimation.')
	parser.add_argument('-f', '--file', metavar='input graph', dest="inputfile")
	parser.add_argument('-m', '--model', metavar='estimation model', dest="model", default="linear1")
	parser.add_argument('-g', '--graph', metavar='random graph type', dest="graph_type", default="scale_free")
	parser.add_argument('-o', '--output', metavar='result file', dest="outputfile", default="results/ate.csv")
	parser.add_argument('-M', '--method', metavar='method', dest="method", default="baseline1")

	args = parser.parse_args()
	config.dynamic["outputfile"] = args.outputfile

	if args.inputfile:
		config.dynamic["community_file"] = "caches/" + args.inputfile.split(".")[0] + "_community.pickle"
		config.dynamic["graph_file"] = "caches/" + args.inputfile.split(".")[0] + "_graph.pickle"
		graph, adjmat = util.load_graph(path=args.inputfile)
	else:
		config.dynamic["community_file"] = "caches/" + args.graph_type + "_community.pickle"
		config.dynamic["graph_file"] = "caches/" + args.graph_type + "_graph.pickle"
		graph, adjmat = util.load_graph(graph_type=args.graph_type)

	print("Starting estimating...")
	true_ate, estimated_ate = estimate(graph, adjmat, args.model, args.method)
	print("Writing result...")
	write_result(args.model, args.method, true_ate, estimated_ate)
	print("Done.")
