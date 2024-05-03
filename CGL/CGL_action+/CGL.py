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
    import pycuda.autoprimaryctx # Fix for HPRC GRACE, autoinit
    import pycuda.autoinit       # Fix for HPRC GRACE

class sim:
    # Default to no state (1D or 2D numpy array for CGoL), side is the side length of the square, seed is used for random state generation, gpu chooses whether to use GPU or not, device selects the GPU device to use.
    def __init__(self, state=None, side=8, seed=8, gpu=False, gpu_select=0, warp=8, spawnStabilityFactor=-1, stableStabilityFactor=1, runBlank=False):
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
        if not isinstance(runBlank, bool):
            raise TypeError('runBlank must be a bool!')
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
            self.initState = np.copy(state)                 # Used when reset() is called.
        else:
            np.random.seed(seed)                            # Set constant seed for random environemnts (useful for debugging).
            self.side = side
            self.size = side ** 2
            if runBlank:                                    # Use a blank grid of zeros rather than random grid.
                self.world = np.zeros(self.size, dtype=np.uint8)
            else:
                self.world = np.random.randint(2, size=self.size, dtype=np.uint8)
            self.initState = np.copy(self.world)            # Used when reset() is called.

        self.max_density = self.get_max_density()                   # See below for functiont to get maximum density for still life.
        self.temp = np.empty_like(self.world)                       # Used here incase of forceCPU=True.
        self.stable = np.zeros(self.size, dtype=np.int8)            # Used to store stable values for each cell, NOTE: IS SIGNED!
        self.stable[self.world != 0] = self.spawnStabilityFactor    # Every cell starts at the spawnStabilityFactor.
        self.initStable = np.copy(self.stable)                      # Needed when reset() is called.

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
            
            # Check that the GPU select option is valid.
            if(gpu_select > device_count - 1):
                gpu_list = range(device_count)
                raise ValueError(f'gpu_select={gpu_select}, however the device which can be chosen are: {gpu_list}.')

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
                    char stabilityValue = stable[cellLoc];
                    stable[cellLoc] = ((currState && prevState) * ((!isMax * (stabilityValue + 1)) | (isMax * stabilityValue))) | ((currState && !prevState) * {});
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
            self.cglStream = cuda.Stream()
        else:
            print('CGL is set to run in CPU mode.')
        print('CGL is now running...')

# --- CONTROL ---
# These are private functions that should not be mixed or used outside of this class.
    # GPU friendly state step.
    def __step_state_gpu(self):
        cuda.memcpy_htod(self.world_gpu, np.ascontiguousarray(self.world)) # Copy world into GPU.
        cuda.memcpy_htod(self.stable_gpu, np.ascontiguousarray(self.stable)) # Copy stability values into GPU.
        self.run_gpu(self.world_gpu, self.result_gpu, self.stable_gpu, np.uint32(self.side), np.uint32(self.size), block=(self.blockSize, 1, 1), grid=(self.gridSize, 1), stream=self.cglStream)  # Launch the kernel.
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
    def reward(self, emptyScale=0, reward_exp=0, curr_density=0, useDensity=False):
        reward = 0
        emptyScaleReward = 0
        densityReward = 0
        base = np.add.reduce(self.stable, dtype=np.int32)

        # Will take from state space and motivate the NN to choose that are empty. Zero out cells that are alive as to not bias them.
        if emptyScale != 0:
            rewardWorld = self.world.astype(np.int8)
            aliveIndx = self.world == 1
            rewardWorld[rewardWorld == 0] = np.int8(self.spawnStabilityFactor * emptyScale)
            rewardWorld[aliveIndx] = 0     
            emptyScaleReward = np.add.reduce(rewardWorld, dtype=np.int32)
        
        # Exponential reward modifier to reduce destorying cells.
        if reward_exp != 0:
            densityReward = base * reward_exp ** -(self.max_density - curr_density)

        # Use density or stability matrix.
        if not useDensity:
            reward = base + emptyScaleReward + densityReward
        else:
            reward = np.int64(self.alive())     # This is done for torch tensors.

        return reward
    
    # Returns the count of alive cells in the system.
    def alive(self):
        return np.add.reduce(self.world, dtype=np.uint32) # Faster than np.sum() as of 7 APR 2024.
    
    # Will reset the enviornment to the original state.
    # When the CGL simulation is created, it already sets the seed for the system.
    def reset(self):
        self.world = np.copy(self.initState)
        self.stable = np.copy(self.initStable)

    # This will compare some input world state with the current state and return true if they match.
    def match(self, terminalState):
        return (self.world==terminalState.flatten()).all()

    # This will return a breakdown or value count for stability factor matrix.
    def breakdown_stable(self):
        unique, counts = np.unique(self.stable, return_counts=True)
        breakdown = np.asarray((unique, counts))
        return breakdown
    
    # This will return a breakdown or value count for the world.
    def breakdown_state(self):
        unique, counts = np.unique(self.world, return_counts=True)
        breakdown = np.asarray((unique, counts))
        return breakdown

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
    
    # Returns the iteration/count the simulation is on. 
    def get_count(self):
        return self.count
    
    # Returns the seed used in the simulation.
    def get_seed(self):
        return self.seed
    
    # Returns the size of the state.
    def get_state_dim(self):
        return self.size

    # Get the state space, this is all the possible states for each cell.
    def get_state_space_dim(self):
        return 2 ** self.size
    
    # Compute the possible maximum density for still life.
    # This is periodic therefore the use of special tables can be used.
    # https://www.sciencedirect.com/science/article/pii/S0004370212000124?ref=pdf_download&fr=RR-2&rr=87d2acb2c840ea2a
    def get_max_density(self):
        density = 0
        table7 = [0, 0, 4, 6, 8, 16, 18, 28, 36, 43, 
               54, 64, 76, 90, 104, 119, 136, 152, 171, 190,
               210, 232, 253, 276, 302, 326, 353, 379, 407, 437, 
               467, 497, 531, 563, 598, 633, 668, 706, 744, 782, 
               824, 864, 907, 949, 993, 1039, 1085, 1132, 1181, 1229,
               1280, 1331, 1382, 1436, 1490, 1545, 1602, 1658, 1717, 1776, 1835]
        thrm6 = [0, 1, 3, 8, 9, 11, 16, 17, 19, 25, 27, 31, 33, 39, 41, 47, 49]

        if self.side <= 60:
            density = table7[self.side]
        elif self.side % 54 in thrm6:
            density = np.floor((self.size / 2) + (17 / 27) * self.side - 2)
        else:
            density = np.floor((self.size / 2) + (17 / 27) * self.side - 1)

        return density

    # Expects a new state the same dimensions and side length of the original state. This is an alternative to toggling specific states on and off.
    def update_state(self, newState, newStability):
        temp = newState.flatten().astype(np.uint8)
        temp2 = newStability.flatten().astype(np.int8)
        if temp.size != self.size or len(temp) != len(self.world):
            raise ValueError(f'The new state must have the same size and side as the original state!\nWas given size={temp.size} and side={side} but was expecting size={self.size} and side={self.side}.')
        else:
            self.world = np.copy(temp)
            self.stable = np.copy(temp2)

    # This will toggle specific states on and off given a list of indexes as indx.
    # Each indx is a cell location as 1D vector.
    # Return True of operation successful else False.
    def toggle_state(self, indx):
        indx = np.array(indx)                                   # This is smart enough to determine if it is an array or one integer.
        if np.all(indx < self.size) and np.all(indx >= 0):      # Check that each element (index) is less than the size of world and is non-negative.
            self.world[indx] = np.logical_not(self.world[indx]) # Toggle the respective cells.
            self.stable[indx] = self.spawnStabilityFactor       # Update stability factors, set everything to -1.
            self.stable[indx] *= self.world[indx]               # Self.world acts like a mask already 0's and 1's. Use this to clear out any dead cells.
        elif indx != self.size:                                 # This action does nothing, this is intended. The size + 1 is the "do nothing" operation output of the NN.
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
