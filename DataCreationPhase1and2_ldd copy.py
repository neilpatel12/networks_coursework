#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 00:55:54 2022

@author: neilpatel
"""

from Code.Networks import Phase 
import numpy as np
from tqdm import tqdm

#%% Largest Degree Distribution

m = 3
N_max_list = [10 ** n for n in [3,5,6]]
N_range_list = [(10**1, 10**3), (10**3, 10**5), (10**5,10**6)]
step_list = [5,5,3]
no_trials_list = [10 ** n for n in [3,2,1]]

for i in range(len(N_max_list)):
    N_max = N_max_list[i]
    N_range = N_range_list[i]
    N_lower = N_range[0]
    N_upper = N_range[1]
    step = step_list[i]
    phase1_k1_dist_given_N_range_all_trials = []
    phase2_k1_dist_given_N_range_all_trials = []
    for no_trials in tqdm(range(1, no_trials_list[i] + 1)): # +1 so naming of no_trials file is nice
        phase = 1
        trial1 = Phase(phase, m, N_max, ldd = np.linspace(N_lower, N_upper, step, dtype = int), dd = True)
        phase1_k1_dist_given_N_range_all_trials.append(trial1.k1_dist)
        phase = 2
        trial2 = Phase(phase, m, N_max, ldd = np.linspace(N_lower, N_upper, step, dtype = int), dd = True)
        phase2_k1_dist_given_N_range_all_trials.append(trial2.k1_dist)

    np.save(f'Code/Data/ldd/Phase1_k1_dist/k1_dist_N_range_{N_lower}_{N_upper}_step_{step}_no_trials_{no_trials}.npy', phase1_k1_dist_given_N_range_all_trials)
    np.save(f'Code/Data/ldd/Phase2_k1_Dist/k1_dist_N_range_{N_lower}_{N_upper}_step_{step}_no_trials_{no_trials}.npy', phase2_k1_dist_given_N_range_all_trials)


#%%

k1_dist_test = np.load(f'Code/Data/ldd/Phase1_k1_dist/k1_dist_N_range_{N_lower}_{N_upper}_step_{step}_no_trials_{no_trials}.npy')
print(k1_dist_test[1])
