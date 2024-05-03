import CGL
import time
import numpy as np
import argparse
import pickle
import uuid
import helper
from collections import deque
from dqn import DQNAgent
#from validate import validation

if __name__ == "__main__":
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
    parser.add_argument("--exp-size", default=1000,type=int, help='Experience replay buffer length')                                            # experience replay buffer length
    parser.add_argument("--exp-gpu", action='store_true', help='Put experience replay buffer on GPU for speed, defaults to main memory/CPU.')   # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int, help='Update frequency of target network.')                                       # update frequency of target network
    parser.add_argument("--gpu-index", default=0, type=int, help='GPU device to select for neural network and CGL enviornment.')                # GPU index
    parser.add_argument("--max-eps-len", default=1000, type=int, help='Maximum length of each episode.')                                        # maximum time of an episode
    parser.add_argument("--net-mul", default=2, type=float, help='Multiplier for hidden layers in neural network.')                             # Multiplier for hidden values in neural network.
    parser.add_argument("--empty-state", default=0, type=int, help='Used to scale reward regarding empty cells.')                               # Used to scalue up or down the impact of empty cells on reward.
    parser.add_argument("--verbose", action='store_true', help='Print the current state of the system heatmap.')                                # Useful for debugging, print the current state of the system.
    parser.add_argument("--spawn", default=-2, type=int, help='Spawn stability factor.')                                                        # Used to determine at what value the cells spawn.
    parser.add_argument("--stable", default=2, type=int, help='Max stability factor.')                                                          # Used to determine when maximum stability is achieved.
    parser.add_argument("--cpu", action='store_true', help='Force CGL to use CPU.')                                                             # Used for non-gpu systems.
    parser.add_argument("--run-blank", action='store_true', help="Initialize a blank enviornment, override's seed for enviornment.")            # Used for debugging.
    parser.add_argument("--reward-exp", default=0, type=float, help='Exponent used to control reward function.')                                # Modifier for reward function.
    parser.add_argument("--eval-period", default=10, type=int, help='Used to control evaluation period for data collection.')                   # Modifier for reward function.
    parser.add_argument("--count-down", default=10000, type=int, help='Used as a maximum limit to wait for the system to stabalize.')           # Used for heatmap evaluation and still life performance generation.
    parser.add_argument("--reward-convergence", action='store_true', help='Only compute reward after convergence.')                             # Evaluate reward after convergence.
    parser.add_argument("--rand-name", action='store_true', help='Used for HPRC applications with same side seed.')                             # HPRC specific parameter.
    #exploration strategy
    parser.add_argument("--epsilon-start", default=1, help='Start value of epsilon.')                                                           # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01, help='End value of epsilon.')                                                            # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.9965, help='Decay value of epsilon.')                                                      # decay value of epsilon
    args = parser.parse_args()

    env = CGL.sim(side=args.side, seed=args.seed, gpu=(not args.cpu), gpu_select=args.gpu_index, spawnStabilityFactor=args.spawn, stableStabilityFactor=args.stable, runBlank=args.run_blank, emptyMul=args.empty_state)
    action_space = (args.side ** 2) * 2 # Times 2 for blocks and single toggles. #(env.get_side() - 1) ** 2

    # Print list of inputs for debugging.
    for arg in vars(args):
        print(f'{arg}\t{getattr(args, arg)}')

    # Print the starting state.
    helper.print_matrix(env.get_stable(), ' ')

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
    moving_window = deque(maxlen=args.eval_period)
    max_density = env.get_max_density()
    all_empty = env.get_state_dim() * args.spawn * 2
    heat_map = helper.heatmap(args.side)

    # Store's data for visualizaitons.
    data_breakdown = []
    data_evals     = []
    data_rewards   = []
    data_heatmaps  = []

    # Main program loop.
    print('Episodes\tRewards\tTime (s)')
    start = time.process_time()             # Start the timer.
    # Each episode.
    for e in range(args.n_episodes):        # Run for some number of episodes.
        env.reset()                         # Reset the enviornment to what it started with originally.
        state = env.get_stable(vector=True, shallow=True)
        
        curr_reward = []
        # Episode duration.
        prev_density = env.alive()
        for _ in range(args.max_eps_len):
            # Learner takes action.
            center = learner.select_action(state, epsilon) 

            # Have the action impact the state.
            toggle_sequence, action_weight = helper.take_action(center, args.side)
            prev_state = env.get_stable(vector=True, shallow=False)
            env.toggle_state(toggle_sequence)    # Commit action.
            env.step()                           # Update the simulator's state.
            
            # Collect the reward and state and teach the DQN to learn.
            n_state = env.get_stable(vector=True, shallow=False)
            n_world = env.get_state(vector=True, shallow=False)

            # Reward after congergence, must use stability matrix instead of state space!
            if args.reward_convergence:
                old = env.get_stable()
                env.step()
                count_down = args.count_down
                while not env.match(old) and count_down:
                        old = env.get_stable()
                        env.step()
                        count_down -= 1

            # Create reward and make the agent learn.
            #print(e)
            #print(f'ACTION, TYPE={center}, {action_weight}')
            #print('STATE=\n', n_world.reshape(10, 10))
            #print('STABLE=\n', n_state.reshape(10, 10))
            #print('PREV_STABLE=\n', prev_state.reshape(10, 10))
            curr_density = env.alive() / env.get_state_dim()

            curr_state_reward = np.sum(n_state)
            prev_state_reward = np.sum(prev_state)
            if curr_state_reward == all_empty and prev_state_reward == all_empty:
                reward = -1000
            if curr_state_reward < prev_state_reward:
                reward = -2000
            if curr_state_reward == prev_state_reward and action_weight == 0:   # Do nothing with a positive reward, action_weight = 0 is special meaning "do nothing".
                reward = curr_state_reward
            else:
                reward = curr_state_reward + (curr_state_reward - prev_state_reward) * action_weight #env.reward(args.empty_scale, args.reward_exp, curr_density, useDensity=False)
            learner.step(state, center, reward, n_state)
            #print('REWARD=', reward)
            #print(np.sum(n_state), np.sum(prev_state), '\n')
           #input()

            # Get next state and current reward.
            state = n_state
            curr_reward.append(reward)
            env.update_state(n_world, state)     # Restore the enviornment with the actual next state.
        
        # Update epsilon and moving window reward.
        moving_window.append(np.mean(curr_reward))
        epsilon = epsilon * epsilon_decay
        heat_map.update(env.get_state())

        # Optional print outs.
        if e % args.eval_period == 0:
            print(f'{e}\t{np.mean(moving_window)}\t{time.process_time() - start}')

            data_breakdown.append(heat_map.breakdown().T)
            data_evals.append(heat_map.evaluate())
            data_rewards.append(np.mean(moving_window))
            data_heatmaps.append(heat_map.get_heatmap())

            if args.verbose:
                helper.print_matrix(heat_map.get_heatmap(), ' ')
                print(heat_map.evaluate())
                helper.print_matrix(heat_map.breakdown(), '\t')
            heat_map.clear()

            start = time.process_time()         # Start the timer again for new episode.

    # Episodes done, final prints for training and final save data.
    print(f'{args.n_episodes}\t{np.mean(moving_window)}\t{time.process_time() - start}')    # Final printout of of episode, mean reward, and time duration.
    data_breakdown.append(heat_map.breakdown().T)
    data_evals.append(heat_map.evaluate())
    data_rewards.append(np.mean(moving_window))
    data_heatmaps.append(heat_map.get_heatmap())

    # Save the data files.
    name = ''
    if args.rand_name: # Random name generation for HRPC applications with same side length.
        name = f'{args.side}_{uuid.uuid4()}'
    else:
        name = f'{args.side}'
    learner.save(name)  # Save the final state of the learner.
    data2save = {'breakdown' : data_breakdown, 'evals' : data_evals, 'rewards' : data_rewards, 'data_heatmaps' : data_heatmaps, 'eval_period' : args.eval_period}
    with open(f'data_{name}.pkl', 'wb') as f:
        pickle.dump(data2save, f)

    # Run validation check.
    #validation(learner.Q)

quit()
