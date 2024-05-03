import numpy as np

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

# Take action between single toggle or 2x2 block.
def take_action(center, side):
    size = side ** 2
    block_toggle_threshold = size
    single_toggle_threshold  = block_toggle_threshold * 2

    action = []
    action_weight = 0
    if center < block_toggle_threshold:   # Place a 2x2 toggle block, anchored in upper left corner.
        # Coordinate system with wrap around.
        x = center % side
        y = center - x
        left = (x + side - 1) % side
        right = (x + 1) % side
        up = (y + size - side) % size
        down = (y + side) % size

        # Place the 2x2 block anchored in upper left corner.
        action.append(x + y)                # "Center" is top left anchor.
        action.append(right + y)            # Directly right from anchor.
        action.append(x + down)             # Directly underneath anchor.
        action.append(right + down)         # Directly diagnoal and down from anchor.
        action_weight = 3
    elif center < single_toggle_threshold:    # Toggle individual cell.
        center -= block_toggle_threshold   # Shift center back down to actual indexes valid within state space.
        action.append(center)
        action_weight = 2
    else:   # Do nothing threshold.
        action.append(size)
        action_weight = 0

    return (action, action_weight)