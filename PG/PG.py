"""
ECEN 743: Reinforcement Learning
Policy Gradient Assignment
Code tested using
    1. gymnasium 0.27.1
    2. box2d-py  2.3.5
    3. pytorch   2.0.0
    4. Python    3.9.12
1 & 2 can be installed using pip install gymnasium[box2d]

General Instructions
1. This code consists of TODO blocks, read them carefully and complete each of the blocks
2. Type your code between the following lines
            ###### TYPE YOUR CODE HERE ######
            #################################
3. The default hyperparameters should be able to solve LunarLander-v2 in the continuous setting
4. It is not necessary to modify the rest of the code for this assignment, feel free to do so if needed.

"""
import CGL
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt


class value_network(nn.Module):
    '''
    Value Network: Designed to take in state as input and give value as output
    Used as a baseline in Policy Gradient (PG) algorithms
    '''
    def __init__(self,state_dim,action_dim):
        '''
            state_dim (int): state dimenssion
        '''
        super(value_network, self).__init__()
        self.l1 = nn.Linear(state_dim, action_dim*2)###Note: Lines 43-45 adjusted to include action_dim*2, not sure if done properly
        self.l2 = nn.Linear(action_dim*2, action_dim*2)
        self.l3 = nn.Linear(action_dim*2, 1)

    def forward(self,state):
        '''
        Input: State
        Output: Value of state
        '''
        v = F.tanh(self.l1(state))
        v = F.tanh(self.l2(v))
        return self.l3(v)


class policy_network(nn.Module):
    '''
    Policy Network: Designed for continous action space, where given a
    state, the network outputs the mean and standard deviation of the action
    '''
    def __init__(self,state_dim,action_dim,log_std = 0.0):
        """
            state_dim (int): state dimenssion
            action_dim (int): action dimenssion
            log_std (float): log of standard deviation (std)
        """
        super(policy_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.l1 = nn.Linear(action_dim-1,action_dim*2)#(state_dim,action_dim*2)###Note: Line 71-74 adjusted to include action_dim*2, and action_dim-1 not sure if done properly
        self.l2 = nn.Linear(action_dim*2,action_dim*2)
        self.mean = nn.Linear(action_dim*2,action_dim)
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)


    def forward(self,state):
        '''
        Input: State
        Output: Mean, log_std and std of action
        '''
        print(state.shape)
        
        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(a))
        a_mean = self.mean(a)
        a_log_std = self.log_std.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        return a_mean, a_log_std, a_std

    def select_action(self, state):#????
        '''
        Input: State
        Output: Sample drawn from a normal disribution with mean and std
        '''
        a_mean, _, a_std = self.forward(state)
        action = torch.normal(a_mean, a_std)
        return action

    def get_log_prob(self, state, action):
        '''
        Input: State, Action
        Output: log probabilities
        '''
        mean, log_std, std = self.forward(state)
        var = std.pow(2)
        log_density = -(action - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)


class PGAgent():
    '''
    An agent that performs different variants of the PG algorithm
    '''
    def __init__(self,
     state_dim,
     action_dim,
     discount,
     lr,
     gpu_index,
     seed,
     arg_sim
     ):
        """
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            discount (float): discount factor
            lr (float): learning rate
            gpu_index (int): GPU used for training
            seed (int): Seed of simulation
            env (str): Name of environment
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.lr = lr
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.arg_sim = arg_sim
        self.seed = seed
        self.policy = policy_network(state_dim,action_dim)
        self.value = value_network(state_dim,action_dim )
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=self.lr)

    def sample_traj(self,batch_size=2000,evaluate = False,recordvideo=False):
        '''
        Input:
            batch_size: minimum batch size needed for update
            evaluate: flag to be set during evaluation
        Output:
            states, actions, rewards,not_dones, episodic reward
        '''
        if recordvideo:
        
            self.policy.to("cpu") #Move network to CPU for sampling
            env = gym.make(args.env,continuous=True,render_mode="human")
            states = []
            actions = []
            rewards = []
            n_dones = []
            curr_reward_list = []
            while len(states) < batch_size:
                state, _ = env.reset(seed=self.seed)
                #env.render()
                curr_reward = 0
                for t in range(1000):
                    state_ten = torch.from_numpy(state).float().unsqueeze(0)
                    with torch.no_grad():
                        if evaluate:
                            action = self.policy(state_ten)[0][0].numpy() # Take mean action during evaluation
                        else:
                            action = self.policy.select_action(state_ten)[0].numpy() # Sample from distribution during training
                    print(action)
                    action = action.argmax().cpu()#.astype(np.float64)
                    n_state,reward,terminated,truncated,_ = env.step(action) # Execute action in the environment
                    done = terminated or truncated
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    n_done = 0 if done else 1
                    n_dones.append(n_done)
                    state = n_state
                    curr_reward += reward
                    if done:
                        break
                curr_reward_list.append(curr_reward)
            if evaluate:
                return np.mean(curr_reward_list)
                
            return 0#states,actions,rewards,n_dones, np.mean(curr_reward_list)
        
        else:
            self.policy.to("cpu") #Move network to CPU for sampling
            #env = gym.make(args.env,continuous=True)
            env=CGL.sim(side=self.arg_sim.side, seed=self.arg_sim.seed, gpu=True, gpu_select=self.arg_sim.gpu_index, spawnStabilityFactor=-2, stableStabilityFactor=2)###CGL added
            states = []
            actions = []
            rewards = []
            n_dones = []
            curr_reward_list = []
            while len(states) < batch_size:
                env.reset()###Note: Line 202-203 adjusted to reset states with CGL
                state = env.get_stable(vector=True, shallow=True)     
                curr_reward = 0
                for t in range(1000):
                    state_ten = torch.from_numpy(state).float().unsqueeze(0)
                    with torch.no_grad():
                        if evaluate:
                            action = self.policy(state_ten)[0][0].numpy() # Take mean action during evaluation
                        else:
                            action = self.policy.select_action(state_ten)[0].numpy() # Sample from distribution during training
                    action = action.argmax().cpu()#.astype(np.float64) ###Not sure how to handle action here because it is a array of values, so I just pick the argmax.
                    
                    #n_state,reward,terminated,truncated,_ = env.step(action) # Execute action in the environment
                    ### Lines 216-221 added for step. 
                    env.toggle_state(action)    # Commit action.
                    env.step()                  # Update the simulator's state.

                    # Collect the reward and state and teach the DQN to learn.
                    n_state = env.get_stable(vector=True, shallow=True)
                    reward = env.reward()
                    ###
                    #done = terminated or truncated
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    n_done = 0 if done else 1
                    n_dones.append(n_done)
                    state = n_state
                    curr_reward += reward
                    if done:
                        break
                curr_reward_list.append(curr_reward)
            if evaluate:
                return np.mean(curr_reward_list)
            return states,actions,rewards,n_dones, np.mean(curr_reward_list)


    


    def update(self,states,actions,rewards,n_dones,update_type='Baseline'):
        '''
        TODO: Complete this block to update the policy using different variants of PG
        Inputs:
            states: list of states
            actions: list of actions
            rewards: list of rewards
            n_dones: list of not dones
            update_type: type of PG algorithm
        Output:
            None
        '''
        self.policy.to(self.device) #Move policy to GPU
        if update_type == "Baseline":
            self.value.to(self.device)  #Move value to GPU
        states_ten = torch.from_numpy(np.stack(states)).to(self.device)   #Convert to tensor and move to GPU
        action_ten = torch.from_numpy(np.stack(actions)).to(self.device)  #Convert to tensor and move to GPU
        rewards_ten = torch.from_numpy(np.stack(rewards)).to(self.device) #Convert to tensor and move to GPU
        n_dones_ten = torch.from_numpy(np.stack(n_dones)).to(self.device) #Convert to tensor and move to GPU

        if update_type == "Rt":
            '''
            TODO: Peform PG using the cumulative discounted reward of the entire trajectory
            1. Compute the discounted reward of each trajectory (rt)
            2. Compute log probabilities using states_ten and action_ten
            3. Compute policy loss and update the policy
            '''
            ###### TYPE YOUR CODE HERE ######
            # Do steps 1-3
            #################################
            #1
            #split up all of the tensors based on a complete trace
            
            states_split  = []
            actions_split = []
            rewards_split = []
            last_complete = 0
            for i, n in enumerate(n_dones_ten):
                if n == 0:
                    states_split.append(states_ten[last_complete:(i+1)])
                    actions_split.append(action_ten[last_complete:(i+1)])
                    rewards_split.append(rewards_ten[last_complete:(i+1)])
                    last_complete = i+1
            
                              
            
            #
            discount = self.discount
            discounted_reward = []
            for rewards_on_trace in rewards_split:
                discount_factors = torch.pow(discount, torch.arange(len(rewards_on_trace), dtype=torch.float32))
                discounted_reward.append(torch.sum( torch.mul(rewards_on_trace , discount_factors)))


            #2
            logprob = []
            for states_on_trace, actions_on_trace in zip(states_split, actions_split):
                logprob.append(self.policy.get_log_prob(states_on_trace, actions_on_trace))
            

            
            #3
            deltavalue = 0
            for lp, dr in zip(logprob, discounted_reward):
                deltavalue += dr * torch.sum(lp)
            
            deltavalue = deltavalue / len(logprob)
            
            self.optimizer_policy.zero_grad()
            
            loss = deltavalue*(-1)
            loss.backward()
            self.optimizer_policy.step()
            
            
            


        if update_type == 'Gt':
            '''
            TODO: Peform PG using reward_to_go
            1. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
            2. gt should be of the same length as rewards_ten
            3. Compute log probabilities using states_ten and action_ten
            4. Compute policy loss and update the policy
            '''
            gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)

            ###### TYPE YOUR CODE HERE ######
            # Compute reward_to_go (gt)
            #################################

            print(rewards_ten[0])

            
            
            #1
            states_split  = []
            actions_split = []
            rewards_split = []
            n_dones_split = []
            last_complete = 0
            for i, n in enumerate(n_dones_ten):
                if n == 0:
                    n_dones_split.append(n_dones_ten[last_complete:(i+1)])
                    states_split.append(states_ten[last_complete:(i+1)])
                    actions_split.append(action_ten[last_complete:(i+1)])
                    rewards_split.append(rewards_ten[last_complete:(i+1)])
                    last_complete = i+1
            
         
                
            
            gt = []
            discount = self.discount
            print(len(rewards_split))
           
            for rewards_on_trace in rewards_split:
                
                trace_size = len(rewards_on_trace)
                
                for i in range(trace_size):
                    gt_rewards_on_trace = rewards_on_trace[i:]
                    discount_factors = torch.pow(discount, torch.arange(len(gt_rewards_on_trace), dtype=torch.float32))
                    gt.append(torch.sum( torch.mul(gt_rewards_on_trace , discount_factors)))

            
            gt = np.array(gt)
            
            
        
            
            gt = (gt - gt.mean()) / gt.std()
                
            
                
                
             #Helps with learning stablity
            #2
            
            assert len(gt) == len(rewards_ten), "len(gt) == len(rewards_ten)"
            
            #3
            ###### TYPE YOUR CODE HERE ######
            # Compute log probabilities and update the policy
            #################################
            logprob = []
            
            logprob = self.policy.get_log_prob(states_ten, action_ten)
            
            
            
            #4
            deltavalue = 0
            
            
            deltavalue = 0
            for g, lp in zip(gt, logprob):
                deltavalue += g*lp
            
            deltavalue = deltavalue / len(rewards_split)
            
            
            self.optimizer_policy.zero_grad()
            
            loss = deltavalue*(-1)
            loss.backward()
            self.optimizer_policy.step()
            
            
        #if update_type == 'Baseline':
        #    '''
        #    TODO: Peform PG using reward_to_go and baseline
        #    1. Compute values of states, this will be used as the baseline
        #    2. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
        #    3. gt should be of the same length as rewards_ten
        #    4. Compute advantages
        #    5. Update the value network to predict gt for each state (L2 norm)
        #    6. Compute log probabilities using states_ten and action_ten
        #    7. Compute policy loss (using advantages) and update the policy
        #    '''
        #    
        #    #1
        #    with torch.no_grad():
        #        values_adv = self.value(states_ten)
        #    
        #    #2
        #    gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)
        #    
        #    discount = self.discount
        #    
        #    
        #    phi_ten = torch.zeros(n_dones_ten.shape[0],1).to(self.device)
        #    counter = 0
        #    for i, r in enumerate(n_dones_ten):
        #        phi_ten[i] = self.discount**counter
        #        counter += 1
        #        if r == 0:
        #            counter =0
        #    
        #   
        #    
        #    
        #    gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)
        #    
        #    for rewards_on_trace in rewards_split:
        #        trace_size = len(rewards_on_trace)
        #        
        #        for i in range(trace_size):
        #            gt_rewards_on_trace = rewards_on_trace[i:]
        #            discount_factors = torch.pow(discount, torch.arange(len(gt_rewards_on_trace), dtype=torch.float32))
        #            gt[i] = (torch.sum( torch.mul(gt_rewards_on_trace , discount_factors)))
            
            
        if update_type == 'Baseline':
            '''
            TODO: Peform PG using reward_to_go and baseline
            1. Compute values of states, this will be used as the baseline
            2. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
            3. gt should be of the same length as rewards_ten
            4. Compute advantages
            5. Update the value network to predict gt for each state (L2 norm)
            6. Compute log probabilities using states_ten and action_ten
            7. Compute policy loss (using advantages) and update the policy
            '''
            
            
            #1
            with torch.no_grad():
                values_adv = self.value(states_ten)
            
            
            #2
            states_split  = []
            actions_split = []
            rewards_split = []
            n_dones_split = []
            last_complete = 0
            for i, n in enumerate(n_dones_ten):
                if n == 0:
                    n_dones_split.append(n_dones_ten[last_complete:(i+1)])
                    states_split.append(states_ten[last_complete:(i+1)])
                    actions_split.append(action_ten[last_complete:(i+1)])
                    rewards_split.append(rewards_ten[last_complete:(i+1)])
                    last_complete = i+1
                    
            gt = torch.zeros(rewards_ten.shape[0],1).to(self.device)
            
            discount = self.discount
            j = 0
            for rewards_on_trace in rewards_split:
                trace_size = len(rewards_on_trace)
                
                for i in range(trace_size):
                    gt_rewards_on_trace = rewards_on_trace[i:]
                    discount_factors = torch.pow(discount, torch.arange(len(gt_rewards_on_trace), dtype=torch.float32))
                    gt[i+j] = (torch.sum( torch.mul(gt_rewards_on_trace , discount_factors)))
                j += trace_size


       
            
            #quit()
            #3
            assert len(gt) == len(rewards_ten), "len(gt) == len(rewards_ten)"
            
            
            
            #4
            advantages = gt - values_adv
            advantages = (advantages - advantages.mean()) / advantages.std()

            
            
            
            ###### TYPE YOUR CODE HERE ######
            # Do steps 5-7
            #################################

            #5
            
            values_adv2 = self.value(states_ten)
            
            loss_val = F.mse_loss(values_adv2, gt)
            loss_val.backward()
            self.optimizer_value.step()
            
            #6
            
            logprob = self.policy.get_log_prob(states_ten, action_ten)
            
            
            
            
            #7



            loss_pol = -1*logprob.squeeze() * advantages.squeeze()
            loss_pol = loss_pol.mean()     
            self.optimizer_policy.zero_grad()

            loss_pol.backward()
            #print(loss_pol.item)
            self.optimizer_policy.step()
            
            
            
            
''' ###Section commented out in favor of moving it to main file.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2")           # Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-iter", default=500, type=int)           # Maximum number of training iterations
    parser.add_argument("--discount", default=0.99)                  # Discount factor
    parser.add_argument("--batch-size", default=5000, type=int)      # Training samples in each batch of training
    parser.add_argument("--lr", default=5e-3,type=float)             # Learning rate
    parser.add_argument("--gpu-index", default=0,type=int)           # GPU index
    parser.add_argument("--algo", default="Gt",type=str)       # PG algorithm type. Baseline/Gt/Rt
    args = parser.parse_args()

    # Making the environment
    env = gym.make(args.env,continuous=True,render_mode="human")

    # Setting seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]



    kwargs = {
        "state_dim":state_dim,
        "action_dim":action_dim,
        "discount":args.discount,
        "lr":args.lr,
        "gpu_index":args.gpu_index,
        "seed":args.seed,
        "env":args.env
    }
    learner = PGAgent(**kwargs) # Creating the PG learning agent

    moving_window = deque(maxlen=10)
    data = []
    pdata = []
    for e in range(args.n_iter):
        
        #Steps of PG algorithm
            #1. Sample environment to gather data using a policy
            #2. Update the policy using the data
            #3. Evaluate the updated policy
            #4. Repeat 1-3
        
        states,actions,rewards,n_dones,train_reward = learner.sample_traj(batch_size=args.batch_size)
        learner.update(states,actions,rewards,n_dones,args.algo)
        eval_reward= learner.sample_traj(evaluate=True)
        moving_window.append(eval_reward)
        print('Training Iteration {} Training Reward: {:.2f} Evaluation Reward: {:.2f} \
        Average Evaluation Reward: {:.2f}'.format(e,train_reward,eval_reward,np.mean(moving_window)))

        """
        TODO: Write code for
        1. Logging and plotting
        2. Rendering the trained agent
        """
        ###### TYPE YOUR CODE HERE ######
        #################################
        data.append((e,np.mean(moving_window)))
        pdata.append(np.mean(moving_window))
        if all(element > 210 for element in pdata[-10:]):
            learner.sample_traj(evaluate=True,recordvideo=True)
            

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
    # Save the plot
    plt.savefig('/home/grads/b/barna_nicholas/ecen743/ECEN743-SP24-HW05/plots/episodic_rewards_plot.png')
    '''
