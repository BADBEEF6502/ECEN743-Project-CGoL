import CGL
import time
import numpy as np
import argparse
import pickle
from collections import deque
from dqn import DQNAgent

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

        # Place the 2x2 block anchored in upper left corner.
        action.append(x + y)                # "Center" is top left anchor.
        action.append(right + y)            # Directly right from anchor.
        action.append(x + down)             # Directly underneath anchor.
        action.append(right + down)         # Directly diagnoal and down from anchor.
    else:   # Do nothing threshold.
        action.append(side ** 2)
    return action

if __name__ == "__main__":
    EVAL_PERIOD = 10    # Make evaluations every 10th episode.
    # HyperParameters For DQN
    parser = argparse.ArgumentParser(
                        prog='CGL RL Agent -- Main Launcher',
                        description="Main.py is the launcher responsible for initializing and training an RL agent to sustain still life and stability in Conway's Game of Life.",
                        epilog='Texas A&M University ECEN743 - Eyobel, Andy, Nick, Nhan')

    parser.add_argument("--side", default=10, type=int, help='Side length of sim enviornment.')                                                 # Side length of the simulator env.
    parser.add_argument("--seed", default=0, type=int, help='Seeds for randomness.')                                                            # Randomness seeds.
    parser.add_argument("--n-episodes", default=2000, type=int, help='Maximum number of training episodes.')                                    # maximum number of training episodes
    parser.add_argument("--batch-size", default=64, type=int, help='Training batch size.')                                                      # training batch size
    parser.add_argument("--discount", default=0.99, help='Discount factor.')                                                                    # discount factor
    parser.add_argument("--lr", default=5e-4, help='Learning rate.')                                                                            # learning rate
    parser.add_argument("--tau", default=0.001, help='Tau is softness parameter for updating the target network.')                              # soft update of target network
    parser.add_argument("--exp-size", default=int(1e5),type=int, help='Experience replay buffer length')                                        # experience replay buffer length
    parser.add_argument("--exp-gpu", action='store_true', help='Put experience replay buffer on GPU for speed, defaults to main memory/CPU.')   # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int, help='Update frequency of target network.')                                       # update frequency of target network
    parser.add_argument("--gpu-index", default=0, type=int, help='GPU device to select for neural network and CGL enviornment.')                # GPU index
    parser.add_argument("--max-esp-len", default=100, type=int, help='Maximum length of each episode.')                                         # maximum time of an episode
    parser.add_argument("--net-mul", default=2, type=float, help='Multiplier for hidden layers in neural network.')                             # Multiplier for hidden values in neural network.
    parser.add_argument("--empty-scale", default=0, type=float, help='Used to scale reward regarding empty cells.')                             # Used to scalue up or down the impact of empty cells on reward.
    parser.add_argument("--verbose", action='store_true', help='Print the current state of the system heatmap.')                                # Useful for debugging, print the current state of the system.
    parser.add_argument("--spawn", default=-2, type=int, help='Spawn stability factor.')                                                        # Used to determine at what value the cells spawn.
    parser.add_argument("--stable", default=2, type=int, help='Max stability factor.')                                                          # Used to determine when maximum stability is achieved.
    parser.add_argument("--cpu", action='store_true', help='Force CGL to use CPU.')                                                             # Used for non-gpu systems.
    parser.add_argument("--run-blank", action='store_true', help="Initialize a blank enviornment, override's seed for enviornment.")            # Used for debugging.
    parser.add_argument("--reward-exp", actino='store_true', help='Exponent used to contrl reward function.')                                   # Modifier for reward function.
    #exploration strategy
    parser.add_argument("--epsilon-start", default=1, help='Start value of epsilon.')                                                           # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01, help='End value of epsilon.')                                                            # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.9965, help='Decay value of epsilon.')                                                      # decay value of epsilon
    args = parser.parse_args()

    env = CGL.sim(side=args.side, seed=args.seed, gpu=(not args.cpu), gpu_select=args.gpu_index, spawnStabilityFactor=args.spawn, stableStabilityFactor=args.stable, runBlank=args.run_blank)
    action_space = (args.side ** 2) * 2 # Times 2 for blocks and single toggles. #(env.get_side() - 1) ** 2

    # Print list of inputs for debugging.
    for arg in vars(args):
        print(f'{arg}\t{getattr(args, arg)}')

    # Print the starting state.
    print_matrix(env.get_state(), ' ')

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
        "seed":         args.seed,
        "net_mul":      args.net_mul
    }

    # Code to begin teaching the agent.
    learner = DQNAgent(**kwargs)
    epsilon = args.epsilon_start 
    epsilon_decay = args.epsilon_decay
    moving_window = deque(maxlen=100)
    max_density = env.get_max_density()
    density_threshold = deque(maxlen=args.max_esp_len)
    density_threshold.extend([0 * args.max_esp_len])
    heat_map = heatmap(args.side)

    # Store's data for visualizaitons.
    data_breakdown = []
    data_evals     = []
    data_rewards   = []

    # Main program loop.
    print('Episodes\tRewards\tTime (s)')
    start = time.process_time()             # Start the timer.
    # Each episode.
    for e in range(args.n_episodes):        # Run for some number of episodes.
        env.reset()                         # Reset the enviornment to what it started with originally.
        state = env.get_stable(vector=True, shallow=True)
        
        curr_reward = 0
        # Episode duration.
        #for _ in range(args.max_esp_len):   # Run for maximum length of 1 episode.
        last_density_threshold = 0
        density_threshold_counter = 0
        while last_density_threshold <= np.mean(density_threshold):
            center = learner.select_action(state, epsilon) 

            toggle_sequence = take_action(center, args.side)
            env.toggle_state(toggle_sequence)    # Commit action.
            env.step()                           # Update the simulator's state.

            # Collect the reward and state and teach the DQN to learn.
            n_state = env.get_stable(vector=True, shallow=True)
            curr_density = env.alive() / env.get_state_dim()
            reward = env.reward(args.empty_scale, curr_density, args.reward_exp)
            learner.step(state, center, reward, n_state)

            state = n_state
            curr_reward += reward

            if density_threshold_counter == args.max_esp_len:
                density_threshold_counter = 0
                last_density_threshold = np.mean(density_threshold)

            density_threshold.append(curr_density)
            density_threshold_counter += 1

        heat_map.update(env.get_state())
        # Update epsilon and moving window reward.
        moving_window.append(curr_reward)
        epsilon = epsilon * epsilon_decay

        # Optional print outs.
        if e % EVAL_PERIOD == 0:
            print(f'{e}\t{np.mean(moving_window)}\t{time.process_time() - start}')
            data_breakdown.append(heat_map.breakdown().T)
            data_evals.append(heat_map.evaluate())
            data_rewards.append(np.mean(moving_window))
            start = time.process_time()         # Start the timer again for new episode.
            if args.verbose:
                print_matrix(heat_map.get_heatmap(), ' ')
                print(heat_map.evaluate())
                print_matrix(heat_map.breakdown(), '\t')
                heat_map.clear()

    # Episodes done, final prints.
    print(f'{args.n_episodes}\t{np.mean(moving_window)}\t{time.process_time() - start}')    # Final printout of of episode, mean reward, and time duration.
    learner.save(f'{args.side}')  # Save the final state of the learner.
    data2save = {'breakdown' : data_breakdown, 'evals' : data_evals, 'rewards' : data_rewards}
    pickle.dump(data2save, open(f'data_{args.side}.pkl', 'wb'))

quit()
