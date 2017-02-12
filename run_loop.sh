#!/usr/bin/env zsh

for i in $(seq 10)
do
	eval "./run.sh -b"
	sleep 90
done

for i in $(seq 10)
do
	eval "./run.sh"
	sleep 90
done
