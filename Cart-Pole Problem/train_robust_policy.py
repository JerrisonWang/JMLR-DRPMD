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

def select_action_inner(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

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


cov_val = env.cov_val
cov = [[cov_val, 0., 0., 0.],
       [0., cov_val, 0., 0.], [0., 0., cov_val, 0.], [0., 0., 0., cov_val]]
inver_cov = np.linalg.inv(cov)


def logGrad_MC_Theta(s, a, s_prime, Theta):
    feature_sa = inner_Feature_state_action(s, a)
    term1 = np.dot(feature_sa.reshape(4, 1), np.dot(
        inver_cov, s_prime).reshape(1, 4))
    term2 = np.dot(feature_sa.reshape(4, 1), np.dot(
        feature_sa.reshape(1, 4), np.dot(Theta, inver_cov)))
    logGrad_Theta = term1 - term2
    return logGrad_Theta


def inner_Feature_state_action(s, a): 
    x, x_dot, theta, theta_dot = s.tolist()
    force = a

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    tau = 0.02
    gravity = 9.8
    temp = (
        force + polemass_length * theta_dot**2 * sintheta
    ) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    x_dot = x_dot + tau * xacc
    x = x + tau * x_dot
    theta_dot = theta_dot + tau * thetaacc
    theta = theta + tau * theta_dot
    f_sa = np.array([x, x_dot, theta, theta_dot])
    return f_sa


FAULT_TIMES = 20
OUTER_CONVERGE_TIMES = 60
INNER_LOOP_MAX = 20
MAX_STEP_IN_EPISODE_outer = 10000
MAX_STEP_IN_EPISODE_inner = 100
BETA_STEP = 0.0000001
THETA_BOUND = np.array([0.001, 0.01, 0.001, 0.01])
theta_c_vector = np.ones(4)
theta_c_matrix = np.diag(theta_c_vector)


def inner_update(theta_inner_matrix):
    for inner_loop_index in range(INNER_LOOP_MAX):

        sample_state_inner = []
        samle_action_inner = []
        sample_reward_inner = []
        sample_step = 0

        state, _ = env.reset()
        sample_state_inner.append(state)
        while sample_step < MAX_STEP_IN_EPISODE_inner:
            sample_step += 1
            action = select_action_inner(state)
            samle_action_inner.append(action)
            state, reward, done, _, _ = env.step(action)
            sample_state_inner.append(state)
            sample_reward_inner.append(reward)

            if done:
                break

        t_loop_max = len(sample_state_inner)
        for t in range(t_loop_max-1): 

            G_sample_cost = sample_reward_inner[t:]
            G_k_t = 0.0
            for n_cal_G in range(len(G_sample_cost)):
                G_k_t += G_sample_cost[n_cal_G]*(args.gamma ** n_cal_G)
            update_step = G_k_t*BETA_STEP*(args.gamma ** t)
            Grad_Theta = logGrad_MC_Theta(
                sample_state_inner[t], samle_action_inner[t], sample_state_inner[t+1], theta_inner_matrix)
            x_theta = theta_inner_matrix - \
                update_step*Grad_Theta
            for i in range(len(theta_inner_matrix[:, 1])):
                left_theta = theta_c_matrix[i, i] - THETA_BOUND[i]
                right_theta = theta_c_matrix[i, i] + THETA_BOUND[i]
                theta_inner_matrix[i, i] = np.minimum(
                    np.maximum(left_theta, x_theta[i, i]), right_theta)
    return theta_inner_matrix
def main():
    theta_inner_vector = np.ones(4)
    theta_inner_matrix = np.diag(theta_inner_vector)
    env.set_theta_inner(theta_inner_matrix)
    ep_reward_discount_list = []
    ep_reward_discount_list_inner = []

    running_reward = 0.0
    solved_times = 0
    discount_gamma = 0.999
    i_episode = 0
    fault_times = 0
    for i_episode in count(1):
        print("i_episode: ",i_episode)
        ep_reward = 0
        ep_reward_discount = 0
        state, _ = env.reset()
        success_flag = True
        for t in range(1, MAX_STEP_IN_EPISODE_outer):
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            ep_reward_discount = ep_reward_discount*args.gamma+reward
            if done:
                fault_times +=1
                success_flag = False
                if fault_times>FAULT_TIMES:
                    print("clear the solved_times")
                    solved_times = 0
                    fault_times = 0
                break
        ep_reward_discount_list.append(ep_reward_discount)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if success_flag:
            solved_times +=1
        if solved_times > OUTER_CONVERGE_TIMES:
            print("Outer Converged")
            break
        finish_episode()
        
        theta_inner_matrix = inner_update(theta_inner_matrix)
        env.set_theta_inner(theta_inner_matrix)

    plt.plot(ep_reward_discount_list)
    plt.xlabel('Episode')
    plt.ylabel('Discounted Reward')
    plt.title('Episode Reward (Discounted)')
    plt.show()
    
    reward_episode_table = pd.DataFrame(np.array(ep_reward_discount_list))
    reward_episode_table.to_csv("big_nn_carpole2"+str(time.time())+".csv", index=False)
    torch.save(policy.state_dict(), 'big_robust_policy_params.pth') 
    print("theta_inner_matrix: ",theta_inner_matrix)

if __name__ == '__main__':
    main()
