#!/bin/bash

# Will run from 10x10 to 100x100 with experience replay buffer in GPU.
MAX=70
STEP=10
for ((i=STEP;i<=MAX;i=i+STEP)); do
	echo "Running N=${i}"
	GPU_CAPABLE=True python -m cProfile -s tottime main.py --exp-gpu --n-episodes 5000 --max-esp-len 2000 --side $i > results_$i.txt
	wait
done
