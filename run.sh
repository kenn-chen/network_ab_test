#!/usr/bin/env zsh

for i in {1..20}
do
				./exp.py -m linear1 -o output1
				./exp.py -m uniform -o output1
done
				
