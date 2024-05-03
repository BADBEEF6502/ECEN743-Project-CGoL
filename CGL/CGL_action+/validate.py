import CGL
import dqn
import torch
import numpy as np
import helper

# Print a pretty state matrix.
def print_matrix(state, joinStr):
    state_str = np.where(state == 1, '*', '-')
    for row in state_str:
        print(joinStr.join(map(str, row)))

#def validation(Q):
# Constants used for processing, must be same as Q network being loaded.
SPAWN_FACTOR      = -2
STABLE_FACTOR     = 2
MAX_EPS_LEN       = 10
NET_MUL           = 2
GPU_INDEX         = 0
SIDE              = 10
CONVERGENCE_LIMIT = 1000
STATE_DIM         = SIDE ** 2
ACTION_DIM        = STATE_DIM * 2
EMPTY_MUL         = 2

# Initalize neural network and enviornment.
device = torch.device('cuda', index=GPU_INDEX) if torch.cuda.is_available() else torch.device('cpu')
Q = dqn.QNetwork(STATE_DIM, ACTION_DIM, NET_MUL).to(device)
Q.load_state_dict(torch.load(f'Q_{SIDE}.pth'))
Q.eval()

env = CGL.sim(side=SIDE, gpu=True, gpu_select=GPU_INDEX, spawnStabilityFactor=SPAWN_FACTOR, stableStabilityFactor=STABLE_FACTOR, runBlank=True, emptyMul=EMPTY_MUL)

# Validation.
print('*** VALIDATION ***')
state = env.get_state(vector=True, shallow=False)
for e in range(MAX_EPS_LEN):

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', e)
    print('--- BEFORE ACTION ---')
    print_matrix(env.get_state(), ' ')

    state = torch.from_numpy(state)
    center = int(Q.forward(state).argmax())

    toggle_sequence, action_weight = helper.take_action(center, SIDE)
    env.toggle_state(toggle_sequence)    # Commit action.

    action_taken = ''
    if center == STATE_DIM:
        action_taken = 'DO NOTHING'
    elif center > STATE_DIM:
        action_taken = f'BLOCK AT {center % (STATE_DIM)}'
    else:
        action_taken = f'TOGGLE AT {center}'
    
    print('Action taken:', action_taken)
    print('--- AFTER ACTION ---')
    print_matrix(env.get_state(), ' ')

    # Process the action and get the next state (remember NN see's stability matrix).
    env.step() # Update the simulator's state.
    state = env.get_state(vector=True, shallow=False)

    print('--- AFTER STEP ---')
    print_matrix(env.get_state(), ' ')

# Validation convergence.
print('--- CONVERGENCE ---')
old = env.get_stable()
env.step()
count_down = CONVERGENCE_LIMIT
while not env.match(old) and count_down:
        old = env.get_stable()
        env.step()
        count_down -= 1
print_matrix(env.get_state(), ' ')

#print(Q.state_dict())