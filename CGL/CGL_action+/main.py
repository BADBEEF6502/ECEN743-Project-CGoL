import CGL
import time
import numpy as np
import argparse
from collections import deque
from dqn import DQNAgent

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

    # Print list of inputs for debugging.
    for arg in vars(args):
        print(f'{arg}\t{getattr(args, arg)}')

    kwargs = {
        "state_dim":    env.get_state_dim(),
        "action_dim":   env.get_action_space_dim(),
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
            action = learner.select_action(state, epsilon) 
            
            env.toggle_state(action)    # Commit action.
            env.step()                  # Update the simulator's state.

            # Collect the reward and state and teach the DQN to learn.
            n_state = env.get_stable(vector=True, shallow=True)
            reward = env.reward()
            learner.step(state, action, reward, n_state)
            
            state = n_state
            curr_reward += reward
            
        moving_window.append(curr_reward)
        epsilon = epsilon * epsilon_decay
        if e % 10 == 0:
            print(f'{e}\t{np.mean(moving_window)}\t{time.process_time() - start}')
            start = time.process_time()         # Start the timer.
    
    # Final prints.
    print(f'{args.n_episodes}\t{np.mean(moving_window)}\t{time.process_time() - start}')    # Final printout of of episode, mean reward, and time duration.
    learner.save(f'{args.side}')  # Save the final state of the learner.
quit()
