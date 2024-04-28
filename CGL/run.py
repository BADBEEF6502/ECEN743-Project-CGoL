import pygame, time, argparse, matplotlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import CGL
from dqn import DQNAgent


pygame.init()



def render_update(env:CGL.sim,render_dict):
    learner = DQNAgent(**kwargs)
    epsilon = 0 
    moving_window = deque(maxlen=100)
    
    for e in range(args.n_episodes):
        env.reset()
        state = env.get_state(vector=True)
        
        curr_reward = 0
        for t in range(args.max_esp_len):
            action = learner.select_action(state,epsilon) 
            
            env.toggle_state(action)
            
            env.step(action)
            n_state     = env.get_state(vector=True)
            reward      = env.reward()
            
            state = n_state
            curr_reward += reward
            
        moving_window.append(curr_reward)
        
        if e % 10 == 0:
            print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, np.mean(moving_window)))
           

# Color in32 is of the form: AARRGGBB (alpha, R, G, B).
# If age is selected, you may see "pulsing" this is expect. This is because as the BB counts up in RRGGBB, it resests to zero and adds 1 to G, etc...
def render(cgl, delay, dim, color=0xFF, showAge=False):
    render_dict =   {
                        ""
                    }
    
    
    
    display = pygame.display.set_mode(dim)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        image = cgl.get_stable().T if showAge else cgl.get_state().T    # Need to transpose to look normal.

        if showAge == False:
            image[image != 0] = color

        surf = pygame.pixelcopy.make_surface(image)
        surf = pygame.transform.scale(surf, dim)
        pygame.display.set_caption(f"Seed={cgl.get_seed():,} Iteration={cgl.get_count():,} Stability={cgl.sum_stable():,} Alive={cgl.sum_state():,}")
        display.blit(surf, (0, 0))
        pygame.display.update()
        #print(cgl.get_stable(), '\n')
        pygame.time.delay(delay)
        
        render_update(cgl,render_dict)

    pygame.quit()
        
        
        
def train_model(env:CGL.sim,kwargs):   
    
    learner = DQNAgent(**kwargs)
    epsilon = args.epsilon_start 
    epsilon_decay = args.epsilon_decay
    moving_window = deque(maxlen=100)
    data = []
    start = time.perf_counter()

    for e in range(args.n_episodes):
        env.reset()
        state = env.get_state(vector=True)
        
        curr_reward = 0
        for t in range(args.max_esp_len):
            action = learner.select_action(state,epsilon) 
            
            
        
            env.toggle_state(action)
            
            env.step(action)
            n_state     = env.get_state(vector=True)
            reward      = env.reward()
            
            #print(reward)
            
            learner.step(state,action,reward,n_state) #To be implemented
            
            state = n_state
            curr_reward += reward
            
        moving_window.append(curr_reward)
        epsilon = epsilon*epsilon_decay
        if e % 10 == 0:
            print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, np.mean(moving_window)))
           

        data.append((e,np.mean(moving_window)))
    episodes = []
    rewards = []
    
    if isinstance(data[0], tuple):
        # Extracting data from list of tuples
        episodes, rewards = zip(*data)

    plt.plot(episodes, rewards, marker='o')
    plt.xlabel('Episode Number')
    plt.ylabel('Episodic Reward')
    plt.title('Episodic Rewards over Episodes')
    plt.grid(True)
    plt.show()
    
    print('Runtime=', time.perf_counter()-start)
    
    
def visualize_model():
    pass
    

if __name__ == "__main__":
    # HyperParameters For DQN
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2")          # Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)              # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-episodes", default=2000, type=int)     # maximum number of training episodes
    parser.add_argument("--batch-size", default=64, type=int)       # training batch size
    parser.add_argument("--discount", default=0.99)                 # discount factor
    parser.add_argument("--lr", default=5e-4)                       # learning rate
    parser.add_argument("--tau", default=0.001)                     # soft update of target network
    parser.add_argument("--max-size", default=int(1e5),type=int)    # experience replay buffer length
    parser.add_argument("--update-freq", default=4, type=int)       # update frequency of target network
    parser.add_argument("--gpu-index", default=0,type=int)          # GPU index
    parser.add_argument("--max-esp-len", default=500, type=int)    # maximum time of an episode
    #exploration strategy
    parser.add_argument("--epsilon-start", default=1)               # start value of epsilon
    parser.add_argument("--epsilon-end", default=0.01)              # end value of epsilon
    parser.add_argument("--epsilon-decay", default=0.9965)#.995           # decay value of epsilon
    parser.add_argument("--run-type", default="visualize")
    args = parser.parse_args()
    
    

    
    


    window_height = window_length = 500
    delay_time = 0 # Milliseconds.
    sim_side_size = 10
    env = CGL.sim(side=sim_side_size, seed=1230, gpu=False, spawnStabilityFactor=-20, stableStabilityFactor=20)

    state_dim = env.size
    action_dim = env.size 

    

    kwargs = {
        "state_dim":state_dim,
        "action_dim":action_dim,
        "discount":args.discount,
        "tau":args.tau,
        "lr":args.lr,
        "update_freq":args.update_freq,
        "max_size":args.max_size,
        "batch_size":args.batch_size,
        "gpu_index":args.gpu_index
    }


    if "train" in args.run_type:
        train_model(env,kwargs)
        
    elif "visualize" in args.run_type:
        render(env, delay=delay_time, dim=(window_height, window_length), showAge=True)

    
