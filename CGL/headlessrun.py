# When you run this: GPU_CAPABLE='True' python3 headlessrun.py
import CGL
import numpy as np
import time

sample = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
                  
cgl = CGL.sim(state=sample, gpu=True, spawnStabilityFactor=-2, stableStabilityFactor=2)
start = time.perf_counter()
RUN_ITERS = 1000000

for _ in range(RUN_ITERS):
    print(_)
    print(cgl.get_stable(),'\n')
    cgl.step()
print('Runtime=', time.perf_counter()-start)
quit()
