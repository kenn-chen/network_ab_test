#!/usr/bin/env zsh

if [ ! -d caches ]; then
	mkdir 'caches'
fi

if [ ! -d logs ]; then
	mkdir 'logs'
fi

if [ ! -d results ]; then
	mkdir 'results'
fi

if [ ! -f logs/gen.log ]; then
	touch 'logs/gen.log'
fi

if [ ! -f logs/error.log ]; then
	touch 'logs/error.log'
fi

./exp.py -g scale_free -o results/temp.csv >>logs/gen.log 2>>logs/error.log &
./exp.py -g wiki-Vote -o results/temp.csv >>logs/gen.log 2>>logs/error.log&
./exp.py -g soc-Epinions1 -o results/temp.csv >>logs/gen.log 2>>logs/error.log&
./exp.py -g soc-Slashdot0811 -o results/temp.csv >>logs/gen.log 2>>logs/error.log&
