# AUTH: Andy Cox V
# DATE: 19 APR 2024
# LANG: Python 3.11.5
# USAG: GPU_CAPABLE="TRUE/FALSE" python3 <script that uses CGL.py>
# DESC: Conway Game of Life Engine with GPU support - https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
# TODO: Currently only supports 1 CUDA capable GPU.

# Can make multiple objects of this kind and assign each object to a different GPU. Therefore, mullti-task different kinds of trials on different GPUs at the same time.

# Conway's Game of Life Overview:
# CGoL is played on a infinently large orthogonal grid. Basically, a rectanglular sized map with a bunch of square cells.
#   - Since the cells are square that means each cell has 8 neigbors.
#   - Cells can be in either "alive" or "dead" states.
#   - Wrapping to the other sides/edges is allowed since the grid is finite.
# 1. If an alive cell has less than 2 neighbors it dies next cycle.
# 2. If an alive cell has 2 or 3 neighbors it lives to next cycle.
# 3. If an alive cell has more than 3 neighbors it dies next cycle.
# 4. If a dead cell has exactly 3 live neighbors it becomes alive next cycle.

# Optimizations:
# http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA/2-Basic-implementation
# https://github.com/tsutof/game_of_life_pycuda/blob/master/lifegame.py
# https://documen.tician.de/pycuda/tutorial.html
# - Get rid of as many "if" statements as possible.
# - To check for alive neighbor cells instead of 8 if statements use a stream of additions.
# - For stable coputaitons to check if cell is alive or not used a binary mask of shifts instead of inverse and add 1.
# - Internally, 1D arrays are used for speed and 2D arrays are used as the perfered option to interface externally.

import os
import numpy as np

# Enviornmental variable whether to load PyCuda/GPU drivers or not.
if 'GPU_CAPABLE' in os.environ:
    GPU_CAPABLE = os.environ['GPU_CAPABLE'].lower()
    if GPU_CAPABLE == 'true':
        GPU_CAPABLE = True
    elif GPU_CAPABLE == 'false':
        GPU_CAPABLE = False
    else:
        raise TypeError('GPU_CAPABLE must either be "TRUE" or "FALSE"!')
else:
    GPU_CAPABLE = True

# Sometimes you may have faulty drivers and need a bypass. This os var is boolean either true or false.
if GPU_CAPABLE:
    import pycuda.driver as cuda
    import pycuda.tools
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

class sim:
    # Default to no state (1D or 2D numpy array for CGoL), side is the side length of the square, seed is used for random state generation, gpu chooses whether to use GPU or not, device selects the GPU device to use.
    def __init__(self, state=None, side=8, seed=8, gpu=False, gpu_select=0, warp=8, spawnStabilityFactor=-1, stableStabilityFactor=1):
        # Validate input.
        if not isinstance(state, np.ndarray) and state is not None and not isinstance(state, list):
            raise TypeError('state variable must be a list or Numpy ndarray!')
        if not isinstance(side, int):
            raise TypeError('side must be integer!')
        if side < 1:
            raise ValueError('side must be positive integer greater than 0!')
        if not isinstance(seed, int):
            raise TypeError('seed must be integer!')
        if seed < 0:
            raise ValueError('seed must be positive integer!')
        if not isinstance(gpu, bool):
            raise TypeError('gpu must be bool!')
        if not isinstance(seed, int):
            raise TypeError('gpu_select must be integer!')
        if seed < 0:
            raise ValueError('gpu_select must be positive integer!')
        if not isinstance(warp, int):
            raise TypeError('warp must be integer!')
        if warp < 0:
            raise ValueError('warp must be positive integer!')
        if not isinstance(spawnStabilityFactor, int):
            raise TypeError('spawnStabilityFactor must be an integer!')
        if not isinstance(stableStabilityFactor, int):
            raise TypeError('stableStabilityFactor must be an integer!')
        if GPU_CAPABLE != gpu:
            raise TypeError(f'the os enviornment variable "GPU_CAPABLE" (defaults to true) is {GPU_CAPABLE} and "gpu" is {gpu}. Both must be equal in value!\nTo launch use: GPU_CAPABLE="true\\false" python3 <script.py>')

        # Update universal state variables.
        self.size = 0
        self.side = 0
        self.count = 0
        self.seed = seed
        self.spawnStabilityFactor = spawnStabilityFactor
        self.stableStabilityFactor = stableStabilityFactor
        self.gpu = gpu

        if state is not None:
            state = state.flatten().astype(np.uint8)        # MUST convert to 1D 8-bit uint array.
            self.size = state.size

            if self.size == 0:
                raise ValueError('state must be size greater than 0!')

            self.side = int(np.sqrt(self.size))             # Side length of square enviornment.
            self.world = np.copy(state)
        else:
            np.random.seed(seed)                            # Set constant seed for random environemnts (useful for debugging).
            self.side = side
            self.size = side ** 2
            self.world = np.random.randint(2, size=self.size, dtype=np.uint8)

        self.temp = np.empty_like(self.world)                       # Used here incase of forceCPU=True.
        self.stable = np.zeros(self.size, dtype=np.int8)           # Used to store stable values for each cell, NOTE: IS SIGNED!
        self.stable[self.world != 0] = self.spawnStabilityFactor    # Every cell starts at the spawnStabilityFactor.

        # Setup CPU and GPU memory components here for speed.
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_select)
            cuda.init()

            device = cuda.Device(gpu_select)
            maxThreadPerBlock = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
            maxGridDim = device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
            warpSize = device.get_attribute(cuda.device_attribute.WARP_SIZE)
            blocks = warpSize * warp
            device_count = cuda.Device.count()
            
            print('Number of devices detected:', device_count)
            print('Device selected:', gpu_select)
            print('\tName:', device.name())
            print('\tCompute capability:', device.compute_capability())
            print('\tTotal memory:', device.total_memory() / 1048576, 'MB')
            print('\tMax threads per block:', maxThreadPerBlock)
            print('\tMax grid dimension:', maxGridDim)
            print('\tWarp size:', warpSize)
            print(f'\tPerfered number of blocks per thread (warp multiple = {warp}):', blocks)

            if blocks > maxThreadPerBlock:
                raise ValueError(f'With a warp multiple of {warp} the perfered number of blocks exceeds the maximum number of blocks {maxThreadPerBlock}!')

            # CUDA code for GPU, life function carries out the next step in CGOL and computes stablility factor for each cell.
            cudaCode = SourceModule("""
            __global__ void run(const unsigned char* world, unsigned char* result, char* stable, const unsigned int side, const unsigned int size)
            {{
                for (unsigned int cellLoc = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
                    cellLoc < size;
                    cellLoc += blockDim.x * gridDim.x)
                {{
                    /* Used to aid in neighbor computation. */
                    unsigned int x = cellLoc % side;
                    unsigned int y = cellLoc - x;
                    unsigned int left = (x + side - 1) % side;
                    unsigned int right = (x + 1) % side;
                    unsigned int up = (y + size - side) % size;
                    unsigned int down = (y + side) % size;
                    
                    /* Compute the alive neighbors for each cell. */
                    unsigned char neighbors = 
                        world[left + up] + world[x + up] + world[right + up] +
                        world[left + y] + world[right + y] + 
                        world[left + down] + world[x + down] + world[right + down];

                    /* Compute the result of the current state and stable factor for each cell. */
                    unsigned char prevState = world[cellLoc];
                    unsigned char currState = (neighbors == 3) || (neighbors == 2 && prevState);
                    result[cellLoc] = currState;
   
                    /* Stability function evaluation.
                        newStabilityScore(cellTransition) =  
                            1. alive --> alive do min(stabilityScore + 1, MAX_VALUE)
                            2. dead  --> alive do MIN_VALUE
                            3. otherwise       do 0 */
                    unsigned char isMax = stable[cellLoc] == {};
                    char val = stable[cellLoc];
                    stable[cellLoc] = ((currState && prevState) * ((!isMax * (val + 1)) | (isMax * val))) | ((currState && !prevState) * {});
                }}
            }}
            """.format(self.stableStabilityFactor, self.spawnStabilityFactor))

            # Originally used inverse and add 1 for mask, but switched to bit shift for efficiency.
            self.run_gpu = cudaCode.get_function('run')

            # Allocate GPU memory for the arrays.
            self.world_gpu = cuda.mem_alloc(self.size)              # Previous (current, t=0) world state.
            self.stable_gpu = cuda.mem_alloc(self.stable.nbytes)    # Stability values for each state.
            self.result_gpu = cuda.mem_alloc(self.size)             # Resulting (t+1) world state.

            # Get block and grid size given the selected device.
            self.blockSize = blocks if maxThreadPerBlock > blocks else maxThreadPerBlock
            self.gridSize = (self.size + self.blockSize - 1) // self.blockSize

# --- CONTROL ---
# These are private functions that should not be mixed or used outside of this class.
    # GPU friendly state step.
    def __step_state_gpu(self):
        cuda.memcpy_htod(self.world_gpu, np.ascontiguousarray(self.world)) # Copy world into GPU.
        cuda.memcpy_htod(self.stable_gpu, np.ascontiguousarray(self.stable)) # Copy stability values into GPU.
        self.run_gpu(self.world_gpu, self.result_gpu, self.stable_gpu, np.uint32(self.side), np.uint32(self.size), block=(self.blockSize, 1, 1), grid=(self.gridSize, 1))  # Launch the kernel.
        cuda.memcpy_dtoh(self.world, self.result_gpu) # Copy the resulting world back to the host.
        cuda.memcpy_dtoh(self.stable, self.stable_gpu) # Copy the new stable values back to the host.

    # CPU friendly state and stability step.
    def __step_state_cpu(self):
        for cellLoc in range(self.size):
            # Used to aid in neighbor computations.
            x = cellLoc % self.side
            y = cellLoc - x
            left = (x + self.side - 1) % self.side
            right = (x + 1) % self.side
            up = (y + self.size - self.side) % self.size
            down = (y + self.side) % self.size
            
            # Evaluate count of neighbors.
            neighbors = \
            self.world[left + up] + self.world[x + up] + self.world[right + up] + \
            self.world[left + y] + self.world[right + y] + \
            self.world[left + down] + self.world[x + down] + self.world[right + down]

            # Update the state t+1 and stable values.
            prevState = self.world[cellLoc]
            currState = (neighbors == 3) or (neighbors == 2 and prevState)
            self.temp[cellLoc] = currState
            # Stability function evaluation.
            #    newStabilityScore(cellTransition) =  
            #        1. alive --> alive do min(stabilityScore + 1, MAX_VALUE)
            #        2. dead  --> alive do MIN_VALUE
            #        3. otherwise       do 0
            if (currState and prevState):
                if (self.stable[cellLoc] != self.stableStabilityFactor):
                    self.stable[cellLoc] += 1
            elif(currState and (not prevState)):
                self.stable[cellLoc] = self.spawnStabilityFactor
            else:
                self.stable[cellLoc] = 0
        self.world = np.copy(self.temp)

# --- SIMULATOR ---
    # Steps the system into state s+1. Updates count, the state, and the stables.
    def step(self, forceCPU=False):
        self.count += 1
        if self.gpu and not forceCPU:
            self.__step_state_gpu()
        else:
            self.__step_state_cpu()

    # Returns the total stable of the system - NOTE: IS SIGNED!
    def sum_stable(self):
        return np.add.reduce(self.stable, dtype=np.int64) # Faster than np.sum() as of 7 APR 2024.
    
    # Returns the count of alive cells in the system.
    def sum_state(self):
        return np.add.reduce(self.world, dtype=np.uint64)  # Faster than np.sum() as of 7 APR 2024.

# --- I/O ---
    # Will either return a vector OR a 2D square matrix of the system. NEED TO DO DEEP COPY!
    def get_state(self, vector=False, shallow=False):
        out = self.world if vector else self.world.reshape((self.side, self.side))
        if shallow:
            return out
        return np.copy(out)

    # Will either return a vector OR a 2D square matrix of the stables of each cell with respect to their position. NEED TO DO DEEP COPY!
    def get_stable(self, vector=False, shallow=False):
        out = self.stable if vector else self.stable.reshape((self.side, self.side))
        if shallow:
            return out
        return np.copy(out)
    
    # Returns the side length of the state.
    def get_side(self):
        return self.side
    
    # Returns the size of the state.
    def get_size(self):
        return self.size
    
    # Returns the iteration/count the simulation is on. 
    def get_count(self):
        return self.count
    
    # Returns the seed used in the simulation.
    def get_seed(self):
        return self.seed

    # Expects a new state the same dimensions and side length of the original state. This is an alternative to toggling specific states on and off.
    def update_state(self, newState, side):
        temp = newState.flatten().astype(np.uint8)
        if temp.size != self.size or self.side != side:
            raise ValueError(f'The new state must have the same size and side as the original state!\nWas given size={temp.size} and side={side} but was expecting size={self.size} and side={self.side}.')
        else:
            self.world = np.copy(temp)

    # This will toggle specific states on and off given a list of indexes as indx.
    # Each indx is a cell location as 1D vector.
    # Return True of operation successful else False.
    def toggle_state(self, indx):
        indx = np.array(indx)
        if np.all(indx < self.size) and np.all(indx > 0):    # Check that each element (index) is less than the size of world and greater than 0.
            self.world[indx] = np.logical_not(self.world[indx])
        else:
            raise ValueError(f'Not all indexes are valid!\nIndexes must be positive and less than the size of the state {self.size}.')

    # Store all of the attributes of the current system.
    # state=None, side=8, seed=8, gpu=False, gpu_select=0, warp=8, spawnStabilityFactor=-1
    def save(self):
        return (self.world, self.stable, self.side, self.count, self.spawnStabilityFactor, self.stableStabilityFactor)

    # Load from memory an exact state setup.
    def load(self, newState, newstable, side, count, spawnStabilityFactor, stableStabilityFactor):

        if not isinstance(newState, np.ndarray) or not isinstance(newstable, np.ndarray):
            raise TypeError('newState and newstable variables must be a Numpy ndarray!')
        if not isinstance(side, int) or not isinstance(count, int):
            raise TypeError('side and count must be integer!')
        if side < 1 or count < 0:
            raise ValueError('side and count must be positive integers and side greater than 0!')
        if not isinstance(spawnStabilityFactor, int):
            raise TypeError('spawnStabilityFactor must be an integer!')
        if not isinstance(stableStabilityFactor, int):
            raise TypeError('stableStabilityFactor must be an integer!')

        self.stableStabilityFactor = stableStabilityFactor
        self.spawnStabilityFactor = spawnStabilityFactor
        self.size = side ** 2
        self.side = side
        self.count = count
        self.world = newState.flatten().astype(np.uint8)
        self.stable = newstable.flatten().astype(np.int8)
