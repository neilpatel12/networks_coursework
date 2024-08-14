#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 00:55:54 2022

@author: neilpatel
"""

from Code.Networks import Phase 
import numpy as np
from tqdm import tqdm

#%% Degree distribution
m_list = [1, 3, 9, 27, 81, 243] #[3 ** n for n in [0,1,2,3,4,5,6]]
no_trials_list = [10000, 1000, 1000, 1000, 100, 100] # [10 ** n for n in [4,3,3,3,2,2]]
N_max = 10000

print(f'N_max = {N_max}')
for i in range(len(m_list)):
    m = m_list[i]
    print(f'm = {m}')
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
        
    np.save(f'Code/Data/dd/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase1_deg_dist_given_m_all_trials)
    np.save(f'Code/Data/dd/Phase1_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase1_k_list_given_m_all_trials)
    
    np.save(f'Code/Data/dd/Phase2_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase2_deg_dist_given_m_all_trials)
    np.save(f'Code/Data/dd/Phase2_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase2_k_list_given_m_all_trials)

#%%
#deg_dist_test = np.load(f'Code/Data/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)

#print(deg_dist_test[1])















