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

methods=('baseline1' 'baseline2' 'baseline3')
graphs=('growing_network' 'wiki-Vote' 'soc-Epinions1' 'soc-Slashdot0811')
for M in $methods
do
	for g in $graphs
	do
		eval "./main.py -M $M -m linear1 -o results/tmp.csv -g $g >>logs/run.log 2>>logs/error.log &"
	done
done
