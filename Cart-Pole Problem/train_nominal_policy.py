import argparse
import gymnasium as gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math
import time
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v1')
torch.manual_seed(args.seed)
env.set_cov(0.00000001)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

ep_reward_discount_list = []
OUTER_CONVERGE_TIMES = 25
def main():
    theta_inner_vector = np.array([1.001,1.001,1.001,1.001])
    theta_inner_matrix = np.diag(theta_inner_vector)
    env.set_theta_inner(theta_inner_matrix) 
    
    running_reward = 0.0
    solved_times = 0
    discount_gamma = 0.999
    i_episode = 0
    for i_episode in range(400):
        ep_reward = 0
        ep_reward_discount = 0
        state, _ = env.reset()
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            ep_reward_discount = ep_reward_discount*discount_gamma+reward
            if done:
                break
        ep_reward_discount_list.append(ep_reward_discount)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            solved_times +=1
            running_reward = 0
        if solved_times > OUTER_CONVERGE_TIMES:
            print("Outer Converged")
    
    torch.save(policy.state_dict(), 'no_robust_policy_params.pth')


if __name__ == '__main__':
    main()
