#!/usr/bin/env zsh

methods=('baseline1' 'baseline2')
models=('uniform' 'linear1')
graphs=('-g scale_free' '-f wiki-Vote.txt' '-f soc-Epinions1.txt' '-f soc-Slashdot0811.txt')
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
