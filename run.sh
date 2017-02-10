#!/usr/bin/env zsh

if [ ! -d results ]; then
	mkdir results
fi

if [ ! -d logs ]; then
	mkdir logs
fi

if [ ! -f logs/run.log ]; then
	touch 'logs/run.log'
fi

if [ ! -f logs/error.log ]; then
	touch 'logs/error.log'
fi

methods=('baseline1' 'baseline2' 'baseline3')
models=('uniform' 'linear1')
graphs=('-g scale_free' '-g wiki-Vote' '-g soc-Epinions1' '-g soc-Slashdot0811')
for M in $methods
do
	for m in $models
	do
		for g in $graphs
		do
			eval "./exp.py -M $M -m $m $g >>logs/run.log 2>>logs/error.log &"
		done
	done
done
