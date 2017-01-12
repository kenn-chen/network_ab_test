#!/usr/bin/env zsh

if [! -d results]; then
	mkdir results
fi

if [! -d logs]; then
	mkdir logs
fi

if [! -f logs/run.log]; then
	touch 'logs/run.log'
fi

methods=('baseline1' 'baseline2')
models=('uniform' 'linear1')
graphs=('-g scale_free' '-f data/wiki-Vote.txt' '-f data/soc-Epinions1.txt' '-f data/soc-Slashdot0811.txt')
for M in $methods
do
	for m in $models
	do
		for g in $graphs
		do
			eval "./exp.py -M $M -m $m $g &>>logs/run.log &"
		done
	done
done
