#!/usr/bin/env zsh

if [[ ! -d results ]]; then
	mkdir results
fi

if [[ ! -d logs ]]; then
	mkdir logs
fi

if [[ ! -f logs/run.log ]]; then
	touch 'logs/run.log'
fi

if [[ ! -f logs/error.log ]]; then
	touch 'logs/error.log'
fi

lambda1=(0 0.25 0.75 1)
lambda2=(0 0.1 0.5 1)
#methods=('baseline1' 'baseline2' 'baseline3' 'new')
methods=('greedybp')
graphs=('growing_network' 'wiki-Vote' 'soc-Epinions1' 'soc-Slashdot0811')
for l1 in $lambda1; do
	for l2 in $lambda2; do

		for M in $methods; do
			for g in $graphs; do
				eval "./main.py -l1 $l1 -l2 $l2 -M $M -m linear1 -g $g $b >>logs/run.log 2>>logs/error.log &"
			done
		done

		sleep 60

	done
done
