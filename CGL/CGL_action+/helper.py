import numpy as np
from collections import deque

# Similar to heat map, but only stores last N states.
class NN_state:
    def __init__(self, side, max):
        self.side = side
        self.size = side ** 2
        self.max = max
        self.buff = deque(maxlen=max)

        for _ in range(self.max):
            self.buff.append(np.copy(np.zeros(self.size, dtype=np.uint8)))
    
    def clear(self):
        self.buff.clear()
        for _ in range(self.max):
            self.buff.append(np.copy(np.zeros(self.size, dtype=np.uint8)))

    def get_state(self):
        result = np.array([])
        for i in range(self.max):
            result = np.append(result, self.buff[i])
        return result
    
    def update(self, state):
        self.buff.appendleft(np.copy(state))

# Similar to heat map, but only stores last N states.
class vizbuff:
    def __init__(self, side, max):
        self.side = side
        self.size = side ** 2
        self.max = max
        self.counter = 0
        self.buff = []
        for i in range(self.max):
            self.buff.append(np.zeros(self.size, dtype=np.uint8))
    
    def clear(self):
        self.buff.clear()
        for i in range(self.max):
            self.buff.append(np.zeros(self.size, dtype=np.uint8))

    def get_viz(self):
        result = np.zeros(self.size, dtype=np.uint8)
        for m in self.buff:
            result += m
        return result.flatten()
    
    def update(self, state):
        if self.counter == self.max:
            self.counter = 0

        self.buff[self.counter] = np.copy(state)
        self.counter += 1

# Heat map used to check placement of life.
class heatmap:
    def __init__(self, side):
        self.heat_map_matrix = np.zeros((side, side), dtype=np.uint32)
    
    def clear(self):
        self.heat_map_matrix = np.zeros_like(self.heat_map_matrix)

    def get_heatmap(self):
        return self.heat_map_matrix
    
    def breakdown(self):
        unique, counts = np.unique(self.heat_map_matrix, return_counts=True)
        breakdown = np.asarray((unique, counts))
        return breakdown
    
    def evaluate(self):
        return np.sum(self.heat_map_matrix, dtype=np.uint32)
    
    def update(self, state):
        self.heat_map_matrix += state

# Print a pretty state matrix.
def print_matrix(state, joinStr):
#    state_str = np.where(state == 1, '$', '-')
    for row in state:
        print(joinStr.join(map(str, row)))

# Determine if a cell will be alive at specified location.
def isAlive(center, side, state):
        size = side ** 2

        x = center % side
        y = center - x
        left = (x + side - 1) % side
        right = (x + 1) % side
        up = (y + size - side) % size
        down = (y + side) % size

        neighbors = \
        state[left + up] + state[x + up] + state[right + up] + \
        state[left + y] + state[right + y] + \
        state[left + down] + state[x + down] + state[right + down]

        # Update the state t+1 and stable values.
        liveState = (neighbors == 3) and (state[center] != 1) # DON'T KYS, REMEMBER ACTIONS ARE TOGGLES! or (neighbors == 2 and state[center])
        return 1 if (liveState and state[center] != 1) else 0   # Only have 1 if 3 live neighbors and DON'T KYS!


# Take action and return neighbors per cell
def take_action(center, side, state):
    size = side ** 2
    action = []
    isGood = False

    if center < size:   # Block toggle threshold.
        size = side ** 2
        # Coordinate system with wrap around.
        x = center % side
        y = center - x
        left = (x + side - 1) % side
        right = (x + 1) % side
        up = (y + size - side) % size
        down = (y + side) % size

        action.append(x + y)                # "Center" is top left anchor.
        action.append(right + y)            # Directly right from anchor.
        action.append(x + down)             # Directly underneath anchor.
        action.append(right + down)         # Directly diagnoal and down from anchor.

        isGood = isAlive(action[0], side, state) + isAlive(action[1], side, state) + isAlive(action[2], side, state) + isAlive(action[3], side, state)
    else:   # Do nothing.
        action.append(size)
        isGood = 5
    return (action, isGood)


# Take action between single toggle or 2x2 block.
# def take_action(center, side):
#     size = side ** 2
#     block_toggle_threshold = size
#     single_toggle_threshold  = block_toggle_threshold * 2

#     action = []
#     action_weight = 0
#     # if center < block_toggle_threshold:   # Place a 2x2 toggle block, anchored in upper left corner.
#     #     # Coordinate system with wrap around.
#     #     x = center % side
#     #     y = center - x
#     #     left = (x + side - 1) % side
#     #     right = (x + 1) % side
#     #     up = (y + size - side) % size
#     #     down = (y + side) % size

#     #     # Place the 2x2 block anchored in upper left corner.
#     #     action.append(x + y)                # "Center" is top left anchor.
#     #     action.append(right + y)            # Directly right from anchor.
#     #     action.append(x + down)             # Directly underneath anchor.
#     #     action.append(right + down)         # Directly diagnoal and down from anchor.
#     #     action_weight = 3
#     # elif center < single_toggle_threshold:    # Toggle individual cell.
#     #      center -= block_toggle_threshold   # Shift center back down to actual indexes valid within state space.
#     #      action.append(center)
#     #      action_weight = 2
#         # Coordinate system with wrap around.
#         # x = center % side
#         # y = center - x
#         # left = (x + side - 1) % side
#         # right = (x + 1) % side
#         # up = (y + size - side) % size
#         # down = (y + side) % size

#         # options = [left + up, x + up, right + up,
#         #           left + y, x + y, right + y,
#         #           left + down, x + down, right + down]
        
#         # # Random selection whether to place or not.

#         # while not action:
#         #     if:
            
#         #     else:   # Pure Random.
#         #         for i in range(len(options)):
#         #             if np.random.rand() > 0.5:
#         #                 action.append(options[i])
    
#     if center < block_toggle_threshold:
#     #Coordinate system with wrap around.
#         x = center % side
#         y = center - x
#         left = (x + side - 1) % side
#         right = (x + 1) % side
#         up = (y + size - side) % size
#         down = (y + side) % size

#         options = [left + up, x + up, right + up,
#                     left + y, x + y, right + y,
#                     left + down, x + down, right + down]
        
#         action.append(x + y)                # "Center" is top left anchor.
#         action.append(right + y)            # Directly right from anchor.
#         action.append(x + down)             # Directly underneath anchor.
#         action.append(right + down)         # Directly diagnoal and down from anchor.

#         # Random selection whether to place or not.
#         # while not action:
#         #     if np.random.rand() < 0.9:    # 2x2 Block.
#         #         action.append(x + y)                # "Center" is top left anchor.
#         #         action.append(right + y)            # Directly right from anchor.
#         #         action.append(x + down)             # Directly underneath anchor.
#         #         action.append(right + down)         # Directly diagnoal and down from anchor.
#         #     else:                         # Pure Random.
#         #         for i in range(len(options)):
#         #             if np.random.rand() > 0.5:
#         #                 action.append(options[i])
#     else:   # Do nothing threshold.
#         action.append(size)
#         action_weight = 0

#     return (action, action_weight)