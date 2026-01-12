import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from mountain_car_bc import collect_human_demos, torchify_demos, train_policy, PolicyNetwork, evaluate_policy


device = torch.device('cpu')


def collect_random_interaction_data(num_iters):
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')
    for _ in range(num_iters):
        obs = env.reset()

        done = False
        while not done:
            a = env.action_space.sample()
            next_obs, reward, done, info = env.step(a)
            state_next_state.append(np.concatenate((obs,next_obs), axis=0))
            actions.append(a)
            obs = next_obs
    env.close()

    return np.array(state_next_state), np.array(actions)




class InvDynamicsNetwork(nn.Module):
    '''
        Neural network with that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        # This network takes 4 inputs: (pos, vel) from s and (pos, vel) from s'
        # and outputs 3 logits corresponding to the three actions (left, coast, right)
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        # Forward pass with ReLU activations on hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_inverse_dynamics(inv_dyn, s_s2_torch, a_torch, num_iters=500, lr=0.01):
    """Train the inverse dynamics model using cross-entropy loss."""
    optimizer = Adam(inv_dyn.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for i in range(num_iters):
        optimizer.zero_grad()
        logits = inv_dyn(s_s2_torch)
        loss = loss_fn(logits, a_torch)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Inverse dynamics training - Iter {i}, Loss: {loss.item():.4f}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 5000, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()


    #collect random interaction data (more episodes = better coverage of state space)
    num_interactions = 20
    print(f"Collecting random interaction data from {num_interactions} episodes...")
    s_s2, acs = collect_random_interaction_data(num_interactions)
    print(f"Collected {len(acs)} state transitions for inverse dynamics training")
    
    #put the data into tensors for feeding into torch
    s_s2_torch = torch.from_numpy(np.array(s_s2)).float().to(device)
    a_torch = torch.from_numpy(np.array(acs)).long().to(device)


    #initialize and train inverse dynamics model
    inv_dyn = InvDynamicsNetwork()
    print("\nTraining inverse dynamics model...")
    train_inverse_dynamics(inv_dyn, s_s2_torch, a_torch, num_iters=5000, lr=0.01)
    print("Inverse dynamics training complete!\n")



    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, acs_true, obs2 = torchify_demos(demos)

    #predict actions
    state_trans = torch.cat((obs, obs2), dim = 1)
    outputs = inv_dyn(state_trans)
    _, acs = torch.max(outputs, 1)

    #train policy using predicted actions for states this should use your train_policy function from your BC implementation
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)

