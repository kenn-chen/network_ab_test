#!/usr/bin/env python

import argparse
import sys

import util
import estimator
import config


parser = argparse.ArgumentParser(description='Generating caches.')
parser.add_argument('-g', '--graph', metavar='graph name', dest="name")
parser.add_argument('-M', '--method', metavar='method', dest="method")
args = parser.parse_args()

if not args.name:
    print("graph not given!", file=sys.stderr)
    sys.exit(0)
graphs = ['growing_network', 'wiki-Vote', 'soc-Slashdot0811', 'cit-HepPh']
if args.name not in graphs:
    print("graph name incorrect!", file=sys.stderr)
    sys.exit(0)
elif args.method:
    if args.method not in ['b1', 'b2', 'TSC', 'LRC']:
        print("method name incorrect!", file=sys.stderr)
        sys.exit(0)
    G = util.load_graph(args.name)
    config.dynamic["graph_name"] = args.name
    estimator._sampling(G, 'linear1', args.method)
else:
	util.load_graph(args.name)
