import argparse
import numpy as np
import time
import os
import CGL
from datetime import datetime

# GPU_CAPABLE=<true/false> python3 bench.py <args>

# Setup command line arguments.
parser = argparse.ArgumentParser(prog='Bench', description='This program is a CGL simulator benchmark. This program defaults for GPU usage.', epilog='Use GPU_CAPABLE=<true/false> python3 bench.py <args>')
parser.add_argument('--spawn', type=int, default=-2, help='Spawn stability factor.')
parser.add_argument('--stable', type=int, default=2, help='Stable stability fator.')
parser.add_argument('--side', type=int, default=200, help='Side length of the enviornment to run.')
parser.add_argument('--seed', type=int, default=0, help='Set the seed for the random test.')
parser.add_argument('--iters', type=int, default=1000000, help='Number of iterations to run benchmark.')
parser.add_argument('--device', type=int, default=0, help='GPU device to select.')
parser.add_argument('--cpu', action='store_true', help='Use CPU.')  # Store false if not present.
parser.add_argument('--blink', action='store_true', help='Will override all parameters except --cpu and will print a blinker in 5x5 enviornment for 256 iterations (-128 to 127).')
args = parser.parse_args()

if args.spawn > args.stable:
    raise ValueError('Spawn values must be strictly less than or equal to stable values!')
    quit()

blinker = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
elapsed = 0
stable = 0
state = 0
start = time.perf_counter() # Start the timer.

# Run the benchmark.
if args.blink == False:     # Run standard.
    cgl = CGL.sim(side=args.side, seed=args.seed, gpu=(not args.cpu), gpu_select=args.device, spawnStabilityFactor=args.spawn, stableStabilityFactor=args.stable)
    for _ in range(args.iters):
        cgl.step()
    elapsed = time.perf_counter() - start
    stable = cgl.reward()
    state = cgl.alive()
else:   # Simple blinker sanity check.
    cgl = CGL.sim(state=blinker, gpu=(not args.cpu), gpu_select=args.device, spawnStabilityFactor=-128, stableStabilityFactor=127)
    for _ in range(256):
        print(_)
        print(cgl.get_stable(),'\n')
        cgl.step()
    elapsed = time.perf_counter() - start
    stable = cgl.reward()
    state = cgl.alive()

# Print results.
print('CGL Benchmark Results')
print(datetime.today().strftime('%d-%m-%Y %H:%M:%S'))
print('---  INPUT  ---')
for arg in vars(args):
    print(f'{arg}\t{getattr(args, arg)}')
print('--- RESULTS ---')
print('Elapsed time seconds:', elapsed)
print('Stability:           ', stable)
print('Life:                ', state)
quit()