#!/bin/bash

windows=( 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 )

for w in "${windows[@]}"
do
	for i in `seq 1 10`
	do
		./sad -wsize $w >> time_${w}.txt
	done
done
