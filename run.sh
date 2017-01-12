#!/usr/bin/env zsh

methods=('baseline1' 'baseline2')
models=('uniform' 'linear1')
graphs=('-g scale_free' '-f wiki-Vote.txt' '-f soc-Epinions1.txt' '-f soc-Slashdot0811.txt')
for i in {1..25}
do
	for M in $methods
	do
		for m in $models
		do
			for g in $graphs
			do
				./exp.py -M $M -m $m $g -o results/result.csv &>>log/run.log &
			done
		done
	done
done
