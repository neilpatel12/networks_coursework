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
if False:
    m_list = [4, 8, 16, 32] #[3 ** n for n in [0,1,2,3,4,5,6]]
    no_trials_list = [100000, 10000, 1000, 100] # [10 ** n for n in [4,3,3,3,2,2]]
    N_max = 10000
    
    print(f'N_max = {N_max}')
    for i in reversed(range(len(m_list))):
        m = m_list[i]
        print(f'm = {m}')
    
        phase3_deg_dist_given_m_all_trials = []
        phase3_k_list_given_m_all_trials = []
        for no_trials in tqdm(range(1, no_trials_list[i]+1)):
            
            phase = 3
            trial3 = Phase(phase, m, N_max, ldd = [], dd = True)
            phase3_deg_dist_given_m_all_trials.append(trial3.deg_dist)
            phase3_k_list_given_m_all_trials.append(trial3.k_list)
            
        np.save(f'Code/Data/dd/Phase3_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase3_deg_dist_given_m_all_trials)
        np.save(f'Code/Data/dd/Phase3_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase3_k_list_given_m_all_trials)
        print(f'N_max = {N_max} is saved')
        
    
#%%

m_list = [4, 8, 16, 32] #[3 ** n for n in [0,1,2,3,4,5,6]]
no_trials_list = [10000, 10000, 1000, 100] # [10 ** n for n in [4,3,3,3,2,2]]
N_max = 10000

i = 0
m = m_list[i]
print(f'm = {m}')

phase3_deg_dist_given_m_all_trials = []
phase3_k_list_given_m_all_trials = []
for no_trials in tqdm(range(1, no_trials_list[i]+1)):
    
    phase = 3
    trial3 = Phase(phase, m, N_max, ldd = [], dd = True)
    phase3_deg_dist_given_m_all_trials.append(trial3.deg_dist)
    phase3_k_list_given_m_all_trials.append(trial3.k_list)
    
np.save(f'Code/Data/dd/Phase3_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase3_deg_dist_given_m_all_trials)
np.save(f'Code/Data/dd/Phase3_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', phase3_k_list_given_m_all_trials)
print(f'N_max = {N_max} is saved')
    
    