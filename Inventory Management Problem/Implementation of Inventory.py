# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:49:03 2023

@author: Wang Qiuhao

project: Inventory Management Problem with Contious State Space and Discrete Action Space
"""
import sys
import copy
import math
sys.path.append("..")
import random
from math import *
import matplotlib.cm as cm
import matplotlib
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from Basic_Function_for_Continuous_Inventory import *
import cvxpy as cp
import pandas as pd
import time
import gurobipy
from tqdm import trange
from matplotlib.pyplot import MultipleLocator
np.set_printoptions(threshold=np.inf)

#%%
##############
# Pre-setted Parameters
##############
Action = np.array([-8, -4, 3, 6])
num_action = 4
cost_gamma = 0.95
theta_kappa = 15

#%%
'''
Inplementation of DRPMD in robust Inventory
'''
MAX_STEP_IN_EPISODE = 80
beta_update = 0.1
ones_theta_inner = np.ones(2)
theta_c = np.array([-2, 3.5])
init_state = 0
step_outer = 0.1
theta_pi_ini = np.zeros(2)
Inner_loop_max = 400
OUTER_MAX_LOOP = 5000
OUTER_SAMPLE_MAX = 5
Num_trials_DRPG = 20
G_k_record_list_robust = np.zeros((Num_trials_DRPG, OUTER_MAX_LOOP))
##############
# outer loop
##############
for outer_trial_robust in trange(Num_trials_DRPG):
    theta_pi_ini = np.ones(2)
    print("Num_trials", outer_trial_robust)
    for outer_loop in trange(OUTER_MAX_LOOP):
        ######################################################################
        # inner loop : begin
        ######################################################################
        theta_inner_ini = np.ones(2)
        for k in range(Inner_loop_max):
            # sample until max_step (or termination?)
            sample_state = [init_state]
            samle_action = []
            sample_cost = []
            sample_step = 0
            state_now =  sample_state[-1] 
            action_now = 0       
            while sample_step < MAX_STEP_IN_EPISODE:
                    pi_s = softmax_policy_s(state_now, Action, theta_pi_ini)
                    action_now = generate_action_from_policy(pi_s, Action)
                    samle_action.append(action_now)
                    state_now = generate_state_from_sa(
                        sample_state[-1], action_now, theta_inner_ini)
                    sample_state.append(state_now)
                    sample_cost.append(cost_sas(sample_state[-2], action_now, sample_state[-1]))
                    sample_step+=1      
            # Store sample total reward
            # interation loop 2: t
            t_loop_max = len(sample_state)
            for t in range(t_loop_max-1):  # sas'
        
                # calculate G
                G_sample_cost = sample_cost[t:]
                G_k_t = 0.0
                for n_cal_G in range(len(G_sample_cost)):
                    G_k_t += G_sample_cost[n_cal_G]*(cost_gamma ** n_cal_G)
        
                # Update
                update_step = G_k_t*beta_update*(cost_gamma ** t)
                
                Grad_theta = logGrad_MC_thetam(
                    sample_state[t], samle_action[t], sample_state[t+1], theta_inner_ini)
                x_theta = theta_inner_ini + update_step*Grad_theta
                left_theta = theta_c - theta_kappa*ones_theta_inner
                right_theta = theta_c + theta_kappa*ones_theta_inner
                
                theta_inner_ini = np.minimum(np.maximum(left_theta, x_theta), right_theta)               
        ######################################################################
        # inner loop : end
        ######################################################################
        G_k_record,outer_grad = outer_Grad_pi_1(theta_inner_ini,theta_pi_ini,Action,cost_gamma,OUTER_SAMPLE_MAX,MAX_STEP_IN_EPISODE)
        G_k_record_list_robust[outer_trial_robust, outer_loop] = G_k_record
        theta_pi_ini = outer_theta_update(theta_pi_ini,step_outer, outer_grad)
ave_Cost_episode_robust = np.mean(G_k_record_list_robust,axis=0)     
# record J_list
Cost_episode_table = pd.DataFrame(G_k_record_list_robust)
Cost_episode_table.to_csv("DRPG with robustness within 20 trials.csv", index=False)
'''
Inplementation of MC-PG in non-robust Inventory --- Comparison
'''
MAX_STEP_IN_EPISODE = 80
beta_update = 0.1
ones_theta_inner = np.ones(2)
theta_c = np.array([-2, 3.5])
init_state = 0
step_outer = 0.1
theta_pi_ini = np.ones(2)
Inner_loop_max = 400
OUTER_MAX_LOOP = 5000
OUTER_SAMPLE_MAX = 5
Num_trials = 20
G_k_record_list_nonrobust = np.zeros((Num_trials, OUTER_MAX_LOOP))
##############
# outer loop
##############
for outer_trial in trange(Num_trials):
    theta_pi_ini = np.ones(2)
    print("Num_trials", outer_trial)
    for outer_loop in trange(OUTER_MAX_LOOP):
        ######################################################################
        # inner loop : begin
        ######################################################################
        theta_inner_ini = np.ones(2)
        for k in range(Inner_loop_max):
            # sample until max_step (or termination?)
            sample_state = [init_state]
            samle_action = []
            sample_cost = []
            sample_step = 0
            state_now =  sample_state[-1] 
            action_now = 0       
            while sample_step < MAX_STEP_IN_EPISODE:
                    pi_s = softmax_policy_s(state_now, Action, theta_pi_ini)
                    action_now = generate_action_from_policy(pi_s, Action)
                    samle_action.append(action_now)
                    state_now = generate_state_from_sa(
                        sample_state[-1], action_now, theta_inner_ini)
                    sample_state.append(state_now)
                    sample_cost.append(cost_sas(sample_state[-2], action_now, sample_state[-1]))
                    sample_step+=1      
            # Store sample total reward
            # interation loop 2: t
            t_loop_max = len(sample_state)
            for t in range(t_loop_max-1):  # sas'
        
                # calculate G
                G_sample_cost = sample_cost[t:]
                G_k_t = 0.0
                for n_cal_G in range(len(G_sample_cost)):
                    G_k_t += G_sample_cost[n_cal_G]*(cost_gamma ** n_cal_G)
        
                # Update
                update_step = G_k_t*beta_update*(cost_gamma ** t)
                
                Grad_theta = logGrad_MC_thetam(
                    sample_state[t], samle_action[t], sample_state[t+1], theta_inner_ini)
                x_theta = theta_inner_ini + update_step*Grad_theta
                left_theta = theta_c - theta_kappa*ones_theta_inner
                right_theta = theta_c + theta_kappa*ones_theta_inner
                
                theta_inner_ini = np.minimum(np.maximum(left_theta, x_theta), right_theta)               
        ######################################################################
        # inner loop : end
        ######################################################################
        G_k_record,outer_grad = outer_Grad_pi_1(theta_inner_ini,theta_pi_ini,Action,cost_gamma,OUTER_SAMPLE_MAX,MAX_STEP_IN_EPISODE)
        G_k_record_list_nonrobust[outer_trial, outer_loop] = G_k_record
        G_k_record1,outer_grad1 = outer_Grad_pi_1(theta_c,theta_pi_ini,Action,cost_gamma,OUTER_SAMPLE_MAX,MAX_STEP_IN_EPISODE)
        theta_pi_ini = outer_theta_update(theta_pi_ini,step_outer, outer_grad1)
ave_Cost_episode_nonrobust = np.mean(G_k_record_list_nonrobust,axis=0)
# record J_list
Cost_episode_table = pd.DataFrame(G_k_record_list_nonrobust)
Cost_episode_table.to_csv("DRPG without robustness within 20 trials.csv", index=False)


#%%
'''
Plot the Comparison Figure
'''
robust = read_csv_tonumpy("DRPG with robustness within 20 trials.csv")
Arr_robust = np.array(robust)
non_robust = read_csv_tonumpy("DRPG without robustness within 20 trials.csv")
Arr_nonrobust = np.array(non_robust)
ave_Cost_episode_nonrobust = np.mean(non_robust,axis=0)
ave_Cost_episode_robust = np.mean(robust,axis=0)
robust_perc_95 = np.zeros(5000)
robust_perc_5 = np.zeros(5000)
for i in range(5000):
    robust_perc_95[i] = np.percentile(Arr_robust[:,i],95,interpolation='midpoint')
    robust_perc_5[i] = np.percentile(Arr_robust[:,i],5,interpolation='midpoint')
nonrobust_perc_95 = np.zeros(5000)
nonrobust_perc_5 = np.zeros(5000)
for i in range(5000):
    nonrobust_perc_95[i] = np.percentile(Arr_nonrobust[:,i],95,interpolation='midpoint')
    nonrobust_perc_5[i] = np.percentile(Arr_nonrobust[:,i],5,interpolation='midpoint')
    
x = np.arange(0,5000,1)
plt.figure(dpi=300)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ay = plt.gca()
y_major_locator = MultipleLocator(50)
ay.yaxis.set_major_locator(y_major_locator)
ay.plot(x, ave_Cost_episode_robust, color='green', label = r'RMCPMD')
ay.fill_between(x, robust_perc_5, robust_perc_95, color='green', alpha=0.2)
ay.plot(x, ave_Cost_episode_nonrobust, color='blue', label = 'Non-robust MC-PG')
ay.fill_between(x, nonrobust_perc_5, nonrobust_perc_95, color='blue', alpha=0.2)
plt.xlabel('Number of iterations', fontdict={ 'size'   : 14})
plt.ylabel(r'$\Phi$($\pi$)', fontdict={ 'size'   : 14})
ay.legend(fontsize = 13)
plt.savefig('JMLR-Inventory test_robust.pdf', bbox_inches='tight')
