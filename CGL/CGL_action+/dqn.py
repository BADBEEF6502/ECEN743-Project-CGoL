import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ExperienceReplay:
    """ 
    Based on the Replay Buffer implementation of TD3 
    Reference: https://github.com/sfujim/TD3/blob/master/utils.py
    """
    def __init__(self, state_dim, action_dim, max_size, batch_size, gpu_index=0, on_gpu=False):
        self.device = torch.device('cuda', index=gpu_index) if on_gpu else torch.device('cpu')
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = torch.zeros((max_size, state_dim), dtype=torch.int8).to(self.device)       # State is stability factor matrix of the system.
        self.action = torch.zeros((max_size, action_dim), dtype=torch.int32).to(self.device)    # Action is the index of the cell to modify.
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.int8).to(self.device)  # Next state is next stability factor matrix of the system.
        self.reward = torch.zeros((max_size, 1), dtype=torch.int32).to(self.device)             # The returned rewards or the total stability factor of the system.
        self.batch_size = batch_size

    # Add data to experience replay buffer.
    def add(self, state, action, reward, next_state):
        state = torch.tensor(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.tensor(next_state)
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Collect data from experience replay buffer.
    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)
        return (self.state[ind].cuda(), self.action[ind].cuda(), self.reward[ind].cuda(), self.next_state[ind].cuda())

class QNetwork(nn.Module):
    """
    Q Network: designed to take state as input and give out Q values of actions as output
    """
    def __init__(self, state_dim, action_dim, net_mul):
        """
            state_dim (int): state dimenssion
            action_dim (int): action dimenssion
        """
        super(QNetwork, self).__init__()
        hidden = int(action_dim * net_mul)
        self.l1 = nn.Linear(state_dim, hidden).half()
        self.l2 = nn.Linear(hidden, hidden).half()
        self.l3 = nn.Linear(hidden, hidden).half()
        self.l4 = nn.Linear(hidden, hidden).half()
        self.l5 = nn.Linear(hidden, action_dim).half()
        
    def forward(self, state):
        # If vanishing gradient, try something differentiable.
        # Leaky ReLU? Parametric ReLU and reduce to float16?
        state = state.to(torch.float16) # OPTIMIZATION: Can change this to float16, but may cause issues and need to adjust network too.
        q = F.tanh(self.l1(state.cuda()))
        q = F.tanh(self.l2(q))
        q = F.tanh(self.l3(q))
        q = F.tanh(self.l4(q))
        return self.l5(q)

class DQNAgent():

    def __init__(self,
     state_dim, 
     action_dim,
     on_gpu,
     discount=0.99,
     tau=1e-3,
     lr=5e-4,
     update_freq=4,
     max_size=int(1e5),
     batch_size=64,
     gpu_index=0,
     seed=0,
     net_mul=2
     ):
        """
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            on_gpu (bool): Used to put experience replay buffer in main memory (slower) or GPU memory (faster but more expensive).
            discount (float): discount factor
            tau (float): used to update q-target
            lr (float): learning rate
            update_freq (int): update frequency of target network
            max_size (int): experience replay buffer size
            batch_size (int): training batch size
            gpu_index (int): GPU used for training
            seed (int): init some random seed for agent.
            net_mul (int): init neural network hidden layer multiplier
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lr = lr
        self.update_freq = update_freq
        self.batch_size = batch_size
        np.random.seed(seed) # Used for consistency so the DQN's actions are consistent.
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')

        # Setting up the NNs
        self.Q         = QNetwork(state_dim, action_dim, net_mul).to(self.device)
        self.Q_target  = QNetwork(state_dim, action_dim, net_mul).to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        # Experience Replay Buffer
        self.memory = ExperienceReplay(state_dim, 1, max_size, self.batch_size, gpu_index, on_gpu)     
        self.t_train = 0
    
    # Add experience to replay buffer, learn, and update target network.
    def step(self, state, action, reward, next_state):
        """
        1. Adds (s,a,r,s') to the experience replay buffer, and updates the networks
        2. Learns when the experience replay buffer has enough samples
        3. Updates target netowork
        """
        self.memory.add(state, action, reward, next_state)       
        self.t_train += 1 
                    
        if self.memory.size > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.discount) #To be implemented
        
        if (self.t_train % self.update_freq) == 0:
            self.target_update(self.Q, self.Q_target, self.tau) #To be implemented 

    # Have the DQN agent select an action.
    def select_action(self, state, epsilon):
        # Epsilon greedy exploration.
        s = torch.tensor(state, dtype=torch.int8) # Must convert numpy arrays to torch friendly tensors. Tho, these are read-only.
        a = self.Q.forward(s).argmax().cpu()
        if np.random.random_sample() < epsilon:
            while a == self.Q.forward(s).argmax().cpu():
                a = np.random.randint(self.action_dim)
        return np.int32(a)                        # Action is the index the agent want's to toggle from dead to alive or visa versa. Torch cannot handle uint32!

    # Have the DQN agent learn.
    def learn(self, experiences, discount):
        #this is the experience relay section
        states, actions, rewards, next_states = experiences
        
        #1 Compute Target Q Values
        with torch.no_grad():
            max_next_Q_values, _ = torch.max(self.Q_target(next_states), dim = 1, keepdim=True)
            target_Q_values = torch.add(rewards, torch.mul(discount, max_next_Q_values)) #rewards + discount * max_next_Q_values

        # Compute Q(s, a) using self.Q.
        Q_values = self.Q(states).gather(1, actions.long())

        # Compute MSE.
        loss = F.mse_loss(Q_values, target_Q_values)

        # Update the network.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_update(self.Q, self.Q_target, self.tau)
        
    # Update the target network.
    def target_update(self, Q, Q_target, tau):
        """
        Update the target network parameters (param_target) using current Q parameters (param_Q)
        Perform the update using tau, this ensures that we do not change the target network drastically
        1. param_target = tau * param_Q + (1 - tau) * param_target
        Input: Q, Q_target, tau
        Return: None
        """ 
        Q_target_state_dict = Q_target.state_dict()
        Q_state_dict = Q.state_dict()
        
        for key in Q_target_state_dict:
            Q_target_state_dict[key] = tau * Q_state_dict[key] + (1-tau) * Q_target_state_dict[key]
        Q_target.load_state_dict(Q_target_state_dict)

    # This saves the model.
    def save(self, name='unnamed'):
        torch.save(self.Q.state_dict(), f'Q_{name}.pth')
        torch.save(self.Q_target.state_dict(), f'Q_target_{name}.pth')
