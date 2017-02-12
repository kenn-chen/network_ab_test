#!/usr/bin/env zsh

if [[ -d logs ]]; then
	rm -rf logs
fi

mkdir logs
touch 'logs/run.log'
touch 'logs/error.log'

if [[ ! -d results ]]; then
	mkdir results
fi

if [[ "$1" == "-b" ]]; then
	b="-b"
	output="results/b_ate.csv"
else
	b=""
	output="results/ate.csv"
fi

methods=('baseline1' 'baseline2' 'baseline3')
models=('uniform' 'linear1')
graphs=('growing_network' 'wiki-Vote' 'soc-Epinions1' 'soc-Slashdot0811')
for M in $methods
do
	for m in $models
	do
		for g in $graphs
		do
			eval "../main.py -M $M -m $m -o $output -g $g $b >>logs/run.log 2>>logs/error.log &"
		done
	done
done
