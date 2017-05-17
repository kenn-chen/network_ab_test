#!/usr/bin/env python

import argparse
import sys

import util
import estimator
import config


parser = argparse.ArgumentParser(description='Generating caches.')
parser.add_argument('-g', '--graph', metavar='graph name', dest="name")
parser.add_argument('-M', '--method', metavar='method', dest="method")
parser.add_argument('-b', '--binary', dest="binary", action='store_true')
args = parser.parse_args()

if not args.name:
	print("graph not given!", file=sys.stderr)
graphs = ['growing_network', 'wiki-Vote', 'soc-Epinions1', 'soc-Slashdot0811']
if args.name not in graphs:
	print("graph name incorrect!", file=sys.stderr)


if args.method:
	if args.method not in ['b1', 'LRC']:
		print("graph name incorrect!", file=sys.stderr)
	G = util.load_graph(args.name)
	config.dynamic["graph_name"] = args.name
	estimator._sampling(G, 'linear1', args.method)

else:
	util.load_graph(args.name)
