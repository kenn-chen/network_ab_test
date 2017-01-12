#!/usr/bin/env python

import argparse
import os.path

import config
import util
from estimator import estimate


def write_result(model, true_ate, estimated_ate):
	outputfile = config.dynamic['outputfile']
	if not os.path.exists(outputfile):
		with open(outputfile, 'w') as fout:
			fout.write("model,true_ate,estimated_ate\n")
			fout.write("%s,%f,%f\n" % (model, true_ate, estimated_ate))
	else:
		with open(outputfile, 'a') as fout:
			fout.write("%s,%f,%f\n" % (model, true_ate, estimated_ate))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Causal inference estimation.')
	parser.add_argument('-f', '--file', metavar='input graph', dest="inputfile")
	parser.add_argument('-m', '--model', metavar='estimation model', dest="model", default="uniform")
	parser.add_argument('-g', '--graph', metavar='random graph type', dest="graph_type", default="barabasi_albert")
	parser.add_argument('-o', '--output', metavar='result file', dest="outputfile", default="result.csv")

	args = parser.parse_args()
	config.dynamic["outputfile"] = args.outputfile

	if args.inputfile:
		config.dynamic["community_file"] = args.inputfile.split(".")[0] + "_community.pickle"
		config.dynamic["graph_file"] = args.inputfile.split(".")[0] + "_graph.pickle"
		graph, adjmat = util.load_graph(path=args.inputfile)
	else:
		config.dynamic["community_file"] = args.graph_type + "_community.pickle"
		config.dynamic["graph_file"] = args.graph_type + "_graph.pickle"
		graph, adjmat = util.load_graph(graph_type=args.graph_type)

	print("Starting estimating...")
	true_ate, estimated_ate = estimate(graph, adjmat, args.model)
	print("Writing result...")
	write_result(args.model, true_ate, estimated_ate)
	print("Done.")
