import argparse
import pandas as pd
import gymnasium as gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange
from datetime import datetime
import csv
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
policy.load_state_dict(torch.load("no_robust_policy_params.pth"))

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()




ep_reward_discount_list = []
OUTER_CONVERGE_TIMES = 3
def main():
    step_dis_level_list = []
    for i_dis_level in range(11):
        dis_level = i_dis_level / 10 
        print("discount level : ",dis_level)
        theta_inner_vector = np.array([0.999,1.01,1.001,0.99])
        dis_inner_vector = (np.ones(4) - theta_inner_vector)*dis_level
        theta_inner_vector = dis_inner_vector * dis_level + theta_inner_vector

        theta_inner_matrix = np.diag(theta_inner_vector)
        env.set_theta_inner(theta_inner_matrix)
        
        running_reward = 0.0
        solved_times = 0
        discount_gamma = 0.999
        i_episode = 0
        success_times = 0
        ep_reward_discount_all = 0
        step_dis_level = []
        for i_episode in trange(1000):
            ep_reward = 0
            ep_reward_discount = 0
            state, _ = env.reset()
            success_flag = True
            for t in range(1, 10000):
                action = select_action(state)
                state, reward, done, _, _ = env.step(action)
                ep_reward += reward
                ep_reward_discount = ep_reward_discount*discount_gamma+reward
                if done:
                    success_flag = False
                    break
            step_dis_level.append(t)
            ep_reward_discount_all+= ep_reward_discount
            ep_reward_discount_list.append(ep_reward_discount)
            if success_flag:
                success_times +=1
        step_dis_level_list.append(step_dis_level)
        print("success times: ", success_times)
        print("Average Reward: ", ep_reward_discount_all/100)
    df = pd.DataFrame(step_dis_level_list).transpose()
    df.columns = [f'dis_level_{i/10}' for i in range(11)]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df.to_csv('nonrobust_steps.csv', index=False)



if __name__ == '__main__':
    main()
