import CGL
import time
import torch
import random
import numpy as np
import argparse
from collections import deque
from PG import PGAgent



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default=5, type=int)             # Side length of the simulator env.
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
    env = CGL.sim(side=args.side, seed=args.seed, gpu=True, gpu_select=args.gpu_index, spawnStabilityFactor=-2, stableStabilityFactor=2)

    # Setting seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.get_state_space_dim()
    action_dim = env.get_action_space_dim()
    print(state_dim,action_dim)


    kwargs = {
        "state_dim":state_dim,
        "action_dim":action_dim,
        "discount":args.discount,
        "lr":args.lr,
        "gpu_index":args.gpu_index,
        "seed":args.seed,
        "arg_sim":args
    }
    learner = PGAgent(**kwargs) # Creating the PG learning agent

    moving_window = deque(maxlen=10)
    data = []
    pdata = []
    for e in range(args.n_iter):
        '''
        Steps of PG algorithm
            1. Sample environment to gather data using a policy
            2. Update the policy using the data
            3. Evaluate the updated policy
            4. Repeat 1-3
        '''
        states,actions,rewards,n_dones,train_reward = learner.sample_traj(batch_size=args.batch_size)
        learner.update(states,actions,rewards,n_dones,args.algo)
        eval_reward= learner.sample_traj(evaluate=True)
        moving_window.append(eval_reward)
        print('Training Iteration {} Training Reward: {:.2f} Evaluation Reward: {:.2f} \
        Average Evaluation Reward: {:.2f}'.format(e,train_reward,eval_reward,np.mean(moving_window)))