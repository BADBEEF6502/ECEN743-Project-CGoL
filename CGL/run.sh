#!/bin/bash

# Will run from 10x10 to 100x100 with experience replay buffer in GPU.
MAX=100
for ((i=10;i<=MAX;i=i+10)); do
	echo "Running N=${i}"
	GPU_CAPABLE=True python -m cProfile -s tottime main.py --exp-gpu --side $i > results_$i.txt
	wait
done
