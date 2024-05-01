import CGL
import time
import numpy as np
import argparse
from collections import deque
from dqn import DQNAgent


# Print a pretty state matrix.
def print_state(state):
    state_str = np.where(state == 1, '$', '-')
    for row in state_str:
        print(" ".join(map(str, row)))

# Take action between single toggle or 2x2 block.
def take_action(center, side):
    size = side ** 2
    single_toggle_threshold = side ** 2
    block_toggle_threshold  = single_toggle_threshold * 2

    action = []
    if center < single_toggle_threshold:    # Toggle individual cell.
        action.append(center)
    elif center < block_toggle_threshold:   # Place a 2x2 toggle block, anchored in upper left corner.

        center -= single_toggle_threshold   # Shift center back down to actual indexes valid within state space.

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
    else:   # Do nothing threshold.
        action.append(side ** 2)
    return action

if __name__ == "__main__":
    # HyperParameters For DQN
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default=10, type=int)             # Side length of the simulator env.
    parser.add_argument("--seed", default=0, type=int)              # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-episodes", default=2000, type=int)     # maximum number of training episodes
    parser.add_argument("--batch-size", default=64, type=int)       # training batch size
    parser.add_argument("--discount", default=0.99)                 # discount factor
    parser.add_argument("--lr", default=5e-4)                       # learning rate
    parser.add_argument("--tau", default=0.001)                     # soft update of target network
    parser.add_argument("--exp-size", default=int(1e5),type=int)    # experience replay buffer length
    parser.add_argument("--exp-gpu", action='store_true')           # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int)       # update frequency of target network
    parser.add_argument("--gpu-index", default=0,type=int)          # GPU index
    parser.add_argument("--max-esp-len", default=1000, type=int)    # maximum time of an episode
    #exploration strategy
    parser.add_argument("--epsilon-start", default=1)               # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01)              # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.9965)          # decay value of epsilon
    args = parser.parse_args()

    env = CGL.sim(side=args.side, seed=args.seed, gpu=True, gpu_select=args.gpu_index, spawnStabilityFactor=-2, stableStabilityFactor=2)
    action_space = (args.side ** 2) * 2 # Times 2 for blocks and  #(env.get_side() - 1) ** 2

    # Print list of inputs for debugging.
    for arg in vars(args):
        print(f'{arg}\t{getattr(args, arg)}')

    kwargs = {
        "state_dim":    env.get_state_dim(),
        "action_dim":   action_space,
        "on_gpu":       args.exp_gpu,
        "discount":     args.discount,
        "tau":          args.tau,
        "lr":           args.lr,
        "update_freq":  args.update_freq,
        "max_size":     args.exp_size,
        "batch_size":   args.batch_size,
        "gpu_index":    args.gpu_index,
        "seed":         args.seed
    }

    # Code to begin teaching the agent.
    learner = DQNAgent(**kwargs)
    epsilon = args.epsilon_start 
    epsilon_decay = args.epsilon_decay
    moving_window = deque(maxlen=100)

    # Main program loop.
    print('Episodes\tRewards\tTime (s)')
    start = time.process_time()             # Start the timer.
    for e in range(args.n_episodes):        # Run for some number of episodes.
        env.reset()                         # Reset the enviornment to what it started with originally.
        state = env.get_stable(vector=True, shallow=True)
        
        curr_reward = 0
        for _ in range(args.max_esp_len):   # Run for maximum length of 1 episode.
            center = learner.select_action(state, epsilon) 
            
            toggle_sequence = take_action(center, args.side)
            env.toggle_state(toggle_sequence)    # Commit action.
            env.step()                  # Update the simulator's state.

            # Collect the reward and state and teach the DQN to learn.
            n_state = env.get_stable(vector=True, shallow=True)
            reward = env.reward()
            learner.step(state, center, reward, n_state)
            
            state = n_state
            curr_reward += reward
            
        moving_window.append(curr_reward)
        epsilon = epsilon * epsilon_decay
        if e % 10 == 0:
            print(f'{e}\t{np.mean(moving_window)}\t{time.process_time() - start}')
            start = time.process_time()         # Start the timer.
            print_state(env.get_state())
    
    # Final prints.
    print(f'{args.n_episodes}\t{np.mean(moving_window)}\t{time.process_time() - start}')    # Final printout of of episode, mean reward, and time duration.
    learner.save(f'{args.side}')  # Save the final state of the learner.
quit()
