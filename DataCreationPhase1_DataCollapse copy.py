#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 01:06:38 2022

@author: neilpatel
"""

from Code.Networks import Phase 
import numpy as np
from tqdm import tqdm

#%% Degree distribution
m = 3#[3 ** n for n in [0,1,2,3,4,5,6]]
no_trials_list = [100000, 10000, 1000, 100, 10] # [10 ** n for n in [4,3,3,3,2,2]]
N_max_list = [100,1000,10000,100000,1000000]

print(f'm = {m}')
for i in range(len(N_max_list)):
    N_max = N_max_list[i]
    print(f'N_max = {N_max}')
    phase1_deg_dist_given_m_all_trials = []
    phase1_k_list_given_m_all_trials = []
    phase2_deg_dist_given_m_all_trials = []
    phase2_k_list_given_m_all_trials = []
    for no_trials in tqdm(range(1, no_trials_list[i]+1)):
        
        #print(f'm = {m}')
        #print(f'no_trials = {no_trials}')
        
        phase = 1
        trial1 = Phase(phase, m, N_max, ldd = [], dd = True)
        phase1_deg_dist_given_m_all_trials.append(trial1.deg_dist)
        phase1_k_list_given_m_all_trials.append(trial1.k_list)
        
        
        phase = 2
        trial2 = Phase(phase, m, N_max, ldd = [], dd = True)
        phase2_deg_dist_given_m_all_trials.append(trial2.deg_dist)
        phase2_k_list_given_m_all_trials.append(trial2.k_list)
        
    np.save(f'Code/Data/dc/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase1_deg_dist_given_m_all_trials)
    np.save(f'Code/Data/dc/Phase1_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase1_k_list_given_m_all_trials)
    
    np.save(f'Code/Data/dc/Phase2_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase2_deg_dist_given_m_all_trials)
    np.save(f'Code/Data/dc/Phase2_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase2_k_list_given_m_all_trials)
#%%
a = np.load('/Users/neilpatel/OneDrive - Imperial College London/Year 3/CandN/Networks Project/Code/Data/dc/Phase1_deg_dist/deg_dist_m_3_N_max_100_no_trials_100.npy', allow_pickle = True)

a[1][:,0]


