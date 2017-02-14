#!/usr/bin/env zsh

# for i in $(seq 10)
# do
# 	eval "./run.sh -b"
# 	sleep $((90*16))
# done

for i in $(seq 10)
do
	eval "./run.sh"
done
