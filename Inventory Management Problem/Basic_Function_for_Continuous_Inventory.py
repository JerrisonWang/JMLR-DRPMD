# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:44:33 2023

@author: Wang Qiuhao

project: RMDP with parameterized transition kernel in a randomized problem
continuous states + 4 actions with inner-MC solving method
"""

import numpy as np
import math
from math import *
import random
import cvxpy as cp
import pandas as pd


#Outer Radial Feature Parameters
out_Fsa1 = np.array([-1.3, 0.1])
out_Fsa2 = np.array([-0.7, 0.5])
#Inner Radial Feature Parameters
in_Fsa1 = np.array([-4, 5])
in_Fsa2 = np.array([-2, 8])

def read_csv_tonumpy(path):
    data = pd.read_csv(path)
    return np.array(data)
          

##################################################
# Create the feature function
##################################################


def new_Feature_state(s, *args):
    dimension = len(args)
    f_s = np.zeros(dimension)
    for j in range(dimension):
        up = - (state[s] - args[j])**2
        down = 2*1**2
        sclar = 1/(sqrt(2*(1**2)*math.pi))
        f_s[j] = sclar*(up/down)
    return f_s        

def new_Feature_state_action(s, a, *args):
    dimension = len(args)
    f_sa = np.zeros(dimension)
    mid = [s,a]
    state_action = np.array(mid)
    for k in range(dimension):
        up = - (np.linalg.norm(state_action - args[k]))**2
        down = 2*10**2
        sclar = 1/(sqrt(2*(2**2)*math.pi))
        f_sa[k] = sclar*(up/down)
    return f_sa

def outer_Feature_state_action(s, a):
    dimension = 2
    f_sa = np.zeros(dimension)
    mid = [s,a]
    state_action = np.array(mid)
    #Frist Element
    up = - (np.linalg.norm(state_action - out_Fsa1))**2
    #down = 2*10**2
    down = 2*10**2
    sclar = 1/(sqrt(2*(2**2)*math.pi))
    f_sa[0] = sclar*(up/down)
    #Second Element
    up = - (np.linalg.norm(state_action - out_Fsa2))**2
    down = 2*10**2
    sclar = 1/(sqrt(2*(2**2)*math.pi))
    f_sa[1] = sclar*(up/down)
    return f_sa   #Two-dimensional Vector


def inner_Feature_state_action(s, a):
    dimension = 2
    f_sa = np.zeros(dimension)
    mid = [s,a]
    state_action = np.array(mid)
    f_sa = np.zeros(dimension)
    #Frist Element
    up = - (np.linalg.norm(state_action - in_Fsa1))**2
    down = 2*10**2
    sclar = 1/(sqrt(2*(2**2)*math.pi))
    f_sa[0] = sclar*(up/down)
    #Second Element
    up = - (np.linalg.norm(state_action - in_Fsa2))**2
    down = 2*5**2
    sclar = 1/(sqrt(2*(2**2)*math.pi))
    f_sa[1] = sclar*(up/down)
    return f_sa   #Two-dimensional Vector


##################################################
# Create softmax policy
##################################################


def softmax_policy_s(s, action_space, theta_pi):
    lenth_action = len(action_space)
    pi_s = np.zeros(lenth_action)
    feature_sa = np.zeros((2, lenth_action))
    down = 0
    for i in range(lenth_action):
        feature_sa[:,i] = outer_Feature_state_action(s, action_space[i])
        down += np.exp(np.dot(theta_pi, feature_sa[:,i]))
    for j in range(lenth_action):
        up = np.exp(np.dot(theta_pi, feature_sa[:,j]))
        pi_s[j] = up/down 
    return pi_s

##################################################
# generate the gradient of the score function
##################################################


def logGrad_MC_thetam(s, a, s_prime, theta_m):
    feature_sa = inner_Feature_state_action(s, a)
    scalar = (s_prime - np.dot(theta_m, feature_sa)) / 1**2
    logGrad_thetam =  scalar*feature_sa
    return logGrad_thetam

def logGrad_outer_theta(s, a, action_space, theta_pi):
    lenth_action = len(action_space)
    feature_sa = outer_Feature_state_action(s, a) 
    term2 = np.zeros(2)
    pi_s = softmax_policy_s(s, action_space, theta_pi)
    for i in range(lenth_action):
        term2 += pi_s[i]*outer_Feature_state_action(s, action_space[i])
    logGrad_theta_pi = feature_sa - term2
    return logGrad_theta_pi  


##################################################
# Generate the action and the state
##################################################


def generate_action_from_policy(pi_s, action_space):
    rand_num = random.random()
    cumulative_sum = 0
    for i, prob in enumerate(pi_s):
        cumulative_sum += prob
        if rand_num <= cumulative_sum:
            return action_space[i]

def generate_state_from_sa(state_now, action_now, thetam):
    feature_sa = inner_Feature_state_action(state_now, action_now)
    mean = np.dot(feature_sa, thetam)
    sigma = 1
    s_prime = np.random.normal(loc=mean, scale=sigma)
    return s_prime


##################################################
# cost function for (s,a,s')
##################################################


def cost_sas(state_now, action_now, state_next):
    cost = state_now + state_next - action_now
    return cost


##################################################
# Two ways to compute outer gradient
##################################################


def outer_Grad_pi_1(theta_inner, theta_pi, action_space, cost_gamma, sample_episode_times, max_episode_step):
    G_k_record = 0.0
    grad_pi = np.array([0.0, 0.0])
    for sample_i in range(sample_episode_times):
        loggrad_episode = np.array([0.0, 0.0])
        init_state = 0
        # sample episode
        sample_state = [init_state]

        samle_action = []
        sample_cost = []
        sample_step = 0
        state_now = sample_state[-1]
        action_now = 0
        while sample_step < max_episode_step:
            pi_s = softmax_policy_s(sample_state[-1], action_space, theta_pi)
            action_now = generate_action_from_policy(pi_s, action_space)
            samle_action.append(action_now)
            state_now = generate_state_from_sa(
                sample_state[-1], action_now, theta_inner)
            sample_state.append(state_now)
            sample_cost.append(
                    cost_sas(sample_state[-2], action_now, sample_state[-1]))
            sample_step += 1

        # calculate sample G_k
        G_k_t = 0.0
        for n_cal_G in range(len(sample_cost)):
            G_k_t += sample_cost[n_cal_G]*(cost_gamma ** n_cal_G)
        # calculate \sum grad_pi * G_k
        G_k_record += G_k_t
        t_loop_max = len(sample_state)
        for t in range(t_loop_max-1):  # sas'
            loggrad_episode = loggrad_episode + logGrad_outer_theta(
                    sample_state[t], samle_action[t], action_space, theta_pi)
        # add together
        grad_pi = grad_pi + G_k_t*loggrad_episode
    G_k_record = G_k_record/sample_episode_times
    grad_pi = grad_pi/sample_episode_times
    return G_k_record, grad_pi

def outer_Grad_pi_2(theta_inner, theta_pi, action_space, cost_gamma, sample_episode_times, max_episode_step):
    G_k_record = 0.0
    grad_pi = np.array([0.0, 0.0])
    for sample_i in range(sample_episode_times):
        grad_episode = np.array([0.0, 0.0])
        init_state = 0
        # sample episode
        sample_state = [init_state]

        samle_action = []
        sample_cost = []
        sample_step = 0
        state_now = sample_state[-1]
        action_now = 0
        while sample_step < max_episode_step:
            pi_s = softmax_policy_s(sample_state[-1], action_space, theta_pi)
            action_now = generate_action_from_policy(pi_s, action_space)
            samle_action.append(action_now)

            state_now = generate_state_from_sa(
                sample_state[-1], action_now, theta_inner)
            sample_state.append(state_now)
            sample_cost.append(
                    cost_sas(sample_state[-2], action_now, sample_state[-1]))
            sample_step += 1
        #print("sample cost: ",sample_cost)
        # calculate \sum grad_pi * G_k
        t_loop_max = len(sample_state)
        for t in range(t_loop_max-1):  # sas'
            # calculate G_k
            G_sample_cost = sample_cost[t:]
            G_k_t = 0.0
            for n_cal_G in range(len(G_sample_cost)):
                G_k_t += G_sample_cost[n_cal_G]*(cost_gamma ** n_cal_G)
            grad_episode = grad_episode + G_k_t * \
                logGrad_outer_theta(
                    sample_state[t], samle_action[t], action_space, theta_pi)
            if t == 0:
                G_k_record += G_k_t
        # add together
        grad_pi = grad_pi + grad_episode
    G_k_record = G_k_record/sample_episode_times
    grad_pi = grad_pi/sample_episode_times
    return G_k_record, grad_pi


##################################################
# Update the outer with the GD
##################################################


def outer_theta_update(theta_pi_old, step, grad):
    theta_pi_new = theta_pi_old - step*grad
    return theta_pi_new
