import CGL
import time
import numpy as np
import argparse
import pickle
import uuid
import helper
from collections import deque
from dqn import DQNAgent
import torch
#from validate import validation

if __name__ == "__main__":
    # HyperParameters For DQN
    parser = argparse.ArgumentParser(
                        prog='CGL RL Agent -- Main Launcher',
                        description="Main.py is the launcher responsible for initializing and training an RL agent to sustain still life and stability in Conway's Game of Life.",
                        epilog='Texas A&M University ECEN743 - Eyobel, Andy, Nick, Nhan')

    parser.add_argument("--side", default=10, type=int, help='Side length of sim enviornment.')                                                 # Side length of the simulator env.
    parser.add_argument("--seed", default=0, type=int, help='Seeds for randomness.')                                                            # Randomness seeds.
    parser.add_argument("--n-episodes", default=8000, type=int, help='Maximum number of training episodes.')                                    # maximum number of training episodes
    parser.add_argument("--batch-size", default=1024, type=int, help='Training batch size.')                                                      # training batch size
    parser.add_argument("--discount", default=0.99, type=float, help='Discount factor.')                                                        # discount factor
    parser.add_argument("--lr", default=0.001, type=float, help='Learning rate.')                                                               # learning rate
    parser.add_argument("--tau", default=0.001, help='Tau is softness parameter for updating the target network.')                              # soft update of target network
    parser.add_argument("--exp-size", default=5000,type=int, help='Experience replay buffer length')                                            # experience replay buffer length
    parser.add_argument("--exp-gpu", action='store_true', help='Put experience replay buffer on GPU for speed, defaults to main memory/CPU.')   # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int, help='Update frequency of target network.')                                       # update frequency of target network
    parser.add_argument("--gpu-index", default=0, type=int, help='GPU device to select for neural network and CGL enviornment.')                # GPU index
    parser.add_argument("--max-eps-len", default=100, type=int, help='Maximum length of each episode.')                                        # maximum time of an episode
    parser.add_argument("--net-mul", default=2, type=float, help='Multiplier for hidden layers in neural network.')                             # Multiplier for hidden values in neural network.
    parser.add_argument("--empty-state", default=0, type=int, help='Used to scale reward regarding empty cells.')                               # Used to scalue up or down the impact of empty cells on reward.
    parser.add_argument("--empty-min", default=-128, type=int, help='Used as floor value for empty cells.')                                     # Used to scalue up or down the impact of empty cells on reward.
    parser.add_argument("--verbose", action='store_true', help='Print the current state of the system heatmap.')                                # Useful for debugging, print the current state of the system.
    parser.add_argument("--spawn", default=-2, type=int, help='Spawn stability factor.')                                                        # Used to determine at what value the cells spawn.
    parser.add_argument("--stable", default=2, type=int, help='Max stability factor.')                                                          # Used to determine when maximum stability is achieved.
    parser.add_argument("--cpu", action='store_true', help='Force CGL to use CPU.')                                                             # Used for non-gpu systems.
    parser.add_argument("--run-blank", action='store_true', help="Initialize a blank enviornment, override's seed for enviornment.")            # Used for debugging.
    parser.add_argument("--reward-exp", default=0, type=float, help='Exponent used to control reward function.')                                # Modifier for reward function.
#    parser.add_argument("--eval-period", default=10, type=int, help='Used to control evaluation period for data collection.')                  # Modifier for reward function.
    parser.add_argument("--count-down", default=10000, type=int, help='Used as a maximum limit to wait for the system to stabalize.')           # Used for heatmap evaluation and still life performance generation.
    parser.add_argument("--reward-convergence", action='store_true', help='Only compute reward after convergence.')                             # Evaluate reward after convergence.
#    parser.add_argument("--name", help='Used for HPRC applications with same side seed.')                                                       # HPRC specific parameter.
    parser.add_argument("--max-buf", default=2, type=int, help='Number of states for the NN to remember.')                                      # Done in an effort to reduce cycling.
    #exploration strategy
    parser.add_argument("--epsilon-start", default=1, type=float, help='Start value of epsilon.')                                                           # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01, type=float, help='End value of epsilon.')                                                            # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.999, type=float, help='Decay value of epsilon.')                                                      # decay value of epsilon
    args = parser.parse_args()

    env = CGL.sim(side=args.side, seed=args.seed, gpu=(not args.cpu), gpu_select=args.gpu_index, spawnStabilityFactor=args.spawn, stableStabilityFactor=args.stable, runBlank=args.run_blank, empty=args.empty_state, empty_min=args.empty_min)
    action_space = (args.side ** 2) # Times 2 for blocks and single toggles. #(env.get_side() - 1) ** 2

    # Print list of inputs for debugging.
    for arg in vars(args):
        print(f'{arg}\t{getattr(args, arg)}')

    # Print the starting state.
    helper.print_matrix(env.get_state(), ' ')

    MAX_BUFF = 2
    kwargs = {
        "state_dim":    env.get_state_dim() * MAX_BUFF,
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
    moving_window = deque(maxlen=args.max_eps_len)
    max_density = env.get_max_density()
    heat_map = helper.heatmap(args.side)

    # Print for debugging.
    epsilon_tmp = epsilon
    for _ in range(args.n_episodes):
        epsilon_tmp *= epsilon_decay
    print(f'After {args.n_episodes} episodes, epsilon will be {epsilon_tmp}.')

    # Store's data for visualizaitons.
    data_breakdown = []
    data_evals     = []
    data_rewards   = []
    data_heatmaps  = []
    data_actions   = []

    # Main program loop.
    print(f'Max density of side {args.side} is {max_density}.')
    print('Episodes\tRewards\tTime (s)')
    start = time.process_time()             # Start the timer.
    # Each episode.
    # curr_reward = np.zeros(args.max_eps_len, dtype=np.int32)
    still_life_count = 0
    still_life_avg = 0
    NN_viz = helper.NN_state(args.side, MAX_BUFF)

    actions = np.zeros(args.side ** 2, dtype=np.uint32)

    #visual = helper.vizbuff(args.side, 5)
    for e in range(args.n_episodes):        # Run for some number of episodes.
        env.fresh(np.random.randint(0, 999999999))                         # Reset the enviornment to what it started with originally.
        #print(env.get_state())
        #env.reset()
        NN_viz.clear()
        #visual.clear()
        state = env.get_state(vector=True, shallow=False)
        
        center = 0
        curr_reward = 0
        eps_counter = 0

        # Episode duration.
        #while env.alive() < int(max_density * 0.8) and eps_counter < args.max_eps_len:
        while eps_counter < args.max_eps_len:
            #visual.update(state)
            # Have the action impact the state.
            #old_vis = visual.get_viz()
            NN_viz.update(state)
            old_viz = NN_viz.get_state()
            center = learner.select_action(old_viz, epsilon) 
            toggle_sequence, isGood = helper.take_action(center, args.side, state)

            actions[center] += 1

            env.toggle_state(toggle_sequence)    # Commit action.
            env.step()                           # Update the simulator's state.

            # Collect the reward and state and teach the DQN to learn.
            n_state = env.get_state(vector=True, shallow=False)         # What the agent will see, this can change between state and stable.
            NN_viz.update(n_state)
            #n_world = env.get_state(vector=True, shallow=False)         # This must ALWAYS be state!
            #visual.update(n_state)

            # Reward after congergence, must use stability matrix instead of state space!
            # Always use stable here, not state!
            # Would need to use get_stable() and zero out in CGL and here all non-alive cells since they will count down to fix!
            # if args.reward_convergence:
            #     old = env.get_stable()
            #     env.step()
            #     convergence_cycles = 0
            #     while not env.match(old) and convergence_cycles < args.count_down:
            #             old = env.get_stable()
            #             env.step()
            #             convergence_cycles += 1

            # Create reward and make the agent learn.
            #curr_density = env.alive() / max_density

            # Evaluate reward.
            #reward = int(-100 * (1 - (env.alive() / max_density)))  # 100 gives 3 integer places of precision.

            #density_reward = 7 * (-1 + (env.alive() / (max_density // 2)))  # 7 is chosen since that is the max reward from our function.

            # if toggle_sequence[0] == (args.side ** 2):
            #     print('do nothing')
            #     break

            reward = (16 * 2 ** (-(isGood - 4)**2) - 9)# + density_reward   # x = 4 is 7, x = 3 is -1, x = 2 is -8, x = 1 and x = 0 is -10.
            #print(isGood, reward)
            #reward = -1
            learner.step(old_viz, center, reward, NN_viz.get_state())
            #print(visual.get_viz().reshape(args.side, args.side))
            curr_reward += reward

            # Get next state.
            state = n_state
            #env.update_state(n_world, env.get_stable())     # Restore the enviornment with the actual next state.
            eps_counter += 1

        cash_out = 0
        # Cashout reward!
        old = env.get_state()
        env.step()
        convergence_cycles = 0
        while not env.match(old) and convergence_cycles < args.count_down:
                old = env.get_state()
                env.step()
                convergence_cycles += 1

        if env.alive() == 0:
            cash_out = -100
        elif env.alive() > 0 and env.match(old):
            still_life_count += 1
            still_life_avg += env.alive()
            cash_out = 100
        else:
            cash_out = 0

        # PERCENT_DENSITY = 0.12
        # if eps_counter == args.max_eps_len or env.alive() == 0:
        #     cash_out = -100
        # elif env.alive() % 4 == 0:
        #     cash_out = -100
        # elif eps_counter < args.max_eps_len and (env.alive() / max_density) > PERCENT_DENSITY:                                    # End state that has converged with high likleyhood still life.
        #     cash_out = 100
        #     still_life_count += 1
        #     still_life_avg += env.alive()
        # elif (env.alive() / max_density) <= PERCENT_DENSITY:
        #     cash_out = 0
        # else:
        #     ValueError('OMG!')

        old_viz = NN_viz.get_state()    # Get the last thing the NN saw.
        NN_viz.update(env.get_state())  # Get the last thing after convergence.
        learner.step(old_viz, center, cash_out, NN_viz.get_state())

        # if np.sum(env.get_state()) == 0:            # Grid zeros out!
        #     cash_out = -np.abs(max_density * args.empty_min)
        # elif convergence_cycles == args.count_down: # Grid never stabalizes, oscillator.
        #     cash_out = -np.abs(max_density * args.empty_min)
        # elif env.match(old):                                       # Convergence with still life.
        #     cash_out = env.alive() * int(np.floor(np.log2(args.side ** 2))) * max_density
        #     still_life_count += 1
        #     still_life_avg += env.alive()
        # else:
        #     ValueError('OMG!')

        # Update epsilon and moving window reward.
        moving_window.append((curr_reward + cash_out) / (eps_counter + 1))
        #moving_window.append(curr_reward / eps_counter)
        epsilon *= epsilon_decay
        heat_map.update(env.get_state())

        # Optional print outs.
        if e % args.max_eps_len == 0 and e != 0:
            print(f'{e}\t{np.mean(moving_window)}\t{time.process_time() - start}\t{epsilon}\t{still_life_count}\t{still_life_avg / (still_life_count + 1e-8)}')

            top = ''
            bot = ''
            for i in range(len(actions)):
                 if actions[i] != 0:
                    top += ' ' + str(i).zfill(4)
                    bot += ' ' + str(actions[i]).zfill(4)

            print(top)
            print(bot)

            data_breakdown.append(heat_map.breakdown().T)
            data_evals.append(heat_map.evaluate())
            data_rewards.append(np.mean(moving_window))
            data_heatmaps.append(heat_map.get_heatmap())
            data_actions.append(actions)

            if args.verbose:
                helper.print_matrix(heat_map.get_heatmap(), ' ')
                print(heat_map.evaluate())
                helper.print_matrix(heat_map.breakdown(), '\t')
            heat_map.clear()
        
            # Reset vars.
            still_life_count = 0
            still_life_avg = 0
            actions = np.zeros_like(actions)
            start = time.process_time()         # Start the timer again for new episode.

    # Episodes done, final prints for training and final save data.
    print(f'{args.n_episodes}\t{np.mean(moving_window)}\t{time.process_time() - start}\t{epsilon}')    # Final printout of of episode, mean reward, and time duration.
    data_breakdown.append(heat_map.breakdown().T)
    data_evals.append(heat_map.evaluate())
    data_rewards.append(np.mean(moving_window))
    data_heatmaps.append(heat_map.get_heatmap())
    data_actions.append(actions)

    # Save the data files.
    name = f'{args.side}_{args.discount}'
    learner.save(name)  # Save the final state of the learner.
    data2save = {'breakdown' : data_breakdown, 'evals' : data_evals, 'rewards' : data_rewards, 'data_heatmaps' : data_heatmaps, 'data_actions' : data_actions, 'eval_period' : args.max_eps_len}
    with open(f'data_{name}.pkl', 'wb') as f:
        pickle.dump(data2save, f)

    # Run validation check.
    #validation(learner.Q)

quit()
