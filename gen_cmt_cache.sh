#!/usr/bin/env zsh

if [ ! -d cache ]; then
	mkdir 'cache'
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

./exp.py -g scale_free -o results/temp.csv &>>logs/gen.log &
./exp.py -f data/wiki-Vote.txt -o results/temp.csv &>>logs/gen.log &
./exp.py -f data/soc-Epinions1.txt -o results/temp.csv &>>logs/gen.log &
./exp.py -f data/soc-Slashdot0811.txt -o results/temp.csv &>>logs/gen.log &
