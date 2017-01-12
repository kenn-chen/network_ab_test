#!/usr/bin/env zsh

./exp.py -g scale_free -o results/temp.csv &>>log/gen.log &
./exp.py -f wiki-Vote.txt -o results/temp.csv &>>log/gen.log &
./exp.py -f soc-Epinions1.txt -o results/temp.csv &>>log/gen.log &
./exp.py -f soc-Slashdot0811.txt -o results/temp.csv &>>log/gen.log &
