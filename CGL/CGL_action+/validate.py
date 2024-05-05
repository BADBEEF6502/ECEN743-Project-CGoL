import CGL
import dqn
import torch
import numpy as np
import helper
import torch
import sys

# Print a pretty state matrix.
def print_matrix(state, joinStr):
    #state_str = np.where(state == 1, '*', '-')
    state_str = np.where(state == 0, '-', (np.where(state == 1, '*', (np.where(state == 2, 'X', '!')))))
    for row in state_str:
        print(joinStr.join(map(str, row)))

#def validation(Q):
# Constants used for processing, must be same as Q network being loaded.
SPAWN_FACTOR      = 0
STABLE_FACTOR     = 100
EMPTY_MUL         = -1
EMPTY_MIN         = -10
MAX_EPS_LEN       = 2000
NET_MUL           = 2
GPU_INDEX         = 0
SIDE              = int(sys.argv[1])
BUFF_MAX          = 2
CONVERGENCE_LIMIT = 2000
STATE_DIM         = (SIDE ** 2) * BUFF_MAX
ACTION_DIM        = SIDE ** 2
NAME              = float(sys.argv[2])

# Initalize neural network and enviornment.
device = torch.device('cuda', index=GPU_INDEX) if torch.cuda.is_available() else torch.device('cpu')
Q = dqn.QNetwork(STATE_DIM, ACTION_DIM, NET_MUL).to(device)
Q.load_state_dict(torch.load(f'Q_{SIDE}_{NAME}.pth'))
Q.eval()

env = CGL.sim(side=SIDE, seed=100, gpu=True, gpu_select=GPU_INDEX, spawnStabilityFactor=SPAWN_FACTOR, stableStabilityFactor=STABLE_FACTOR, runBlank=True, empty=EMPTY_MUL, empty_min=EMPTY_MIN)

viz = helper.NN_state(SIDE, BUFF_MAX)

actions = np.zeros(SIDE ** 2, dtype=np.uint32)
tot_actions = np.zeros(SIDE **2, dtype=np.uint32)
# Validation.
print('*** VALIDATION ***')


MAX_TRIALS = 100
life_counts = 0
place_final_no_overlap = 0
tot_place_final_no_overlap = 0
tot_perfect = np.zeros(5, dtype=np.uint32)
for i in range(MAX_TRIALS):
    actions = np.zeros_like(actions)
    if i != 0:  # First one is always blank, after that is random.
        env.update_state(np.random.randint(2, size=SIDE**2, dtype=np.uint8))
    state = env.get_state(vector=True, shallow=False)

    print('--- FIRST STATE SAW ---')
    print_matrix(env.get_state(), ' ')

    tot_place_final_no_overlap += place_final_no_overlap
    last_state_saw = 0
    place_final_overlap = 0
    place_final_no_overlap = 0
    for e in range(MAX_EPS_LEN):

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', e)
        print('--- BEFORE ACTION ---')
        print_matrix(env.get_state(), ' ')

        state = torch.from_numpy(viz.get_state())
        #center = int(Q.forward(state).argmax())

        top_vals = 3
        _, top_centers = torch.topk(Q.forward(state), k=top_vals)
        
        # THIS CHOOSES THE ACTIONS, EITHER RANDOM OR LEARNED FROM OUR MODEL!
        center = int(top_centers[np.random.randint(top_vals)])
        #center = np.random.randint(0, SIDE ** 2)

        actions[center] += 1

        toggle_sequence, action_weight = helper.take_action(center, SIDE, state)
        last_state_saw = env.get_state(vector=True, shallow=False)
        env.toggle_state(toggle_sequence)    # Commit action.

        action_taken = ''
        if center == STATE_DIM:
            action_taken = 'DO NOTHING'
        elif center > STATE_DIM:
            action_taken = f'TOGGLE AT {center % (STATE_DIM)}'
        else:
            action_taken = f'BLOCK AT {center}'
        
        print('Action taken:', action_taken)
        print('--- AFTER ACTION ---')
        print_matrix(env.get_state(), ' ')

        # Process the action and get the next state (remember NN see's stability matrix).
        env.step() # Update the simulator's state.
        state = env.get_state(vector=True, shallow=False)
        viz.update(state)

        print('--- AFTER STEP ---')
        print_matrix(env.get_state(), ' ')

    # Validation math.
    for indx in toggle_sequence:
        if last_state_saw[indx] == 1:
            last_state_saw[indx] = 3
            place_final_overlap += 1
        else:
            last_state_saw[indx] = 2
            place_final_no_overlap += 1
    
    tot_perfect[place_final_no_overlap] += 1

    # Validation convergence.
    print('--- LAST STATE SAW ---')

    print_matrix(last_state_saw.reshape(SIDE, SIDE), ' ')
    print('--- CONVERGENCE ---')
    old = env.get_state()
    env.step()
    count_down = CONVERGENCE_LIMIT
    while not env.match(old) and count_down:
            old = env.get_state()
            env.step()
            count_down -= 1
    print_matrix(env.get_state(), ' ')

    if env.alive() != 0:
        life_counts += 1

    top = ''
    bot = ''
    for a in range(len(actions)):
        if actions[a] != 0:
            top += ' ' + str(a).zfill(4)
            bot += ' ' + str(actions[a]).zfill(4)

    print(f'{i} - Final Life:{env.alive()} - Placed Overlapping: {place_final_overlap} - Place Non-Overlapping: {place_final_no_overlap}')
    print(top)
    print(bot)

    tot_actions += actions

print('--- VALIDATION SUMMARY ---')
print('TOTAL TRIALS', MAX_TRIALS)
print('PERCENT CONVERGE WITH LIFE=', (100 * life_counts / MAX_TRIALS))
print('PERCENT FINAL PERFECT BOXES NO OVERLAP=', (100 * tot_perfect[4] / MAX_TRIALS))
print('PERCENT FINAL BOXES ONE OVERLAP=', (100 * tot_perfect[3] / MAX_TRIALS))
print('PERCENT FINAL BOXES TWO OVERLAP=', (100 * tot_perfect[2] / MAX_TRIALS))
print('PERCENT FINAL BOXES THREE OVERLAP=', (100 * tot_perfect[1] / MAX_TRIALS))
print('PERCENT FINAL BOXES TOTAL OVERLAP=', (100 * tot_perfect[0] / MAX_TRIALS))
print('INDIVIDUAL POINTS PLACED WITH NO OVERLAP=', tot_place_final_no_overlap)
print('--- CUMULATIVE ACTIONS ---')
actions_dict = {}
for a in range(len(actions)):
    actions_dict[a] = tot_actions[a]
actions_sorted = dict(sorted(actions_dict.items(), key=lambda item: item[1]))
for key, value in zip(actions_sorted.keys(), actions_sorted.values()):
    print(f"{key} {value}")

print(Q.state_dict())
