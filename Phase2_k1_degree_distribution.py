#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 00:25:08 2022

@author: neilpatel
"""
import numpy as np
import matplotlib.pyplot as plt

params = {
   'axes.labelsize': 24,
   'font.size': 18,
   'font.family': 'sans-serif', # Optionally change the font family to sans-serif
   'font.serif': 'Arial', # Optionally change the font to Arial
   'legend.fontsize': 14,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18, 
   'figure.figsize': [15, 8.8/1.618] # Using the golden ratio and standard column width of a journal
} 

plt.rcParams.update(params)

def largest_deg_dist_p2(N,m):
    N = np.linspace(min(N), max(N), 10000)
    k1 = m - (np.log(N)/(np.log(m) - np.log(m+1)))
    return k1,  N


m = 3
N_max_list = [10 ** n for n in [3,5,6]]
N_range_list = [(10**1, 10**3), (10**3, 10**5), (10**5,10**6)]
step_list = [5,5,3]
no_trials_list = [10 ** n for n in [3,2,1]]



N_lower = 10**1
N_upper = 10**3
step = 5
no_trials = 10**3
set1 = np.load(f'Code/Data/ldd/Phase2_k1_dist/k1_dist_N_range_{N_lower}_{N_upper}_step_{step}_no_trials_{no_trials}.npy')

N_lower = 10**3
N_upper = 10**5
step = 5
no_trials = 10**2
set2 = np.load(f'Code/Data/ldd/Phase2_k1_dist/k1_dist_N_range_{N_lower}_{N_upper}_step_{step}_no_trials_{no_trials}.npy')

N_lower = 10**5
N_upper = 10**6
step = 3
no_trials = 10**1
set3 = np.load(f'Code/Data/ldd/Phase2_k1_dist/k1_dist_N_range_{N_lower}_{N_upper}_step_{step}_no_trials_{no_trials}.npy')

#%%
set1_mean = np.mean(set1, axis = 0)
set2_mean = np.mean(set2, axis = 0)
set3_mean = np.mean(set3, axis = 0)

set1_std = np.std(set1, axis = 0)/ 10**3
set2_std = np.std(set2, axis = 0)/ 10 **2
set3_std = np.std(set3, axis = 0)/ 10

#%%

N_list = np.concatenate((set1_mean[:-1,0], set2_mean[:-1,0], set3_mean[:,0]))
k1_mean_list = np.concatenate((set1_mean[:-1,1], set2_mean[:-1,1], set3_mean[:,1]))
error_list = np.concatenate((set1_std[:-1,1], set2_std[:-1,1], set3_std[:,1]))

#%% actual report plot
#RA_k1_dist
plt.figure()
plt.errorbar(N_list, k1_mean_list, yerr = error_list, marker = 'o', capsize = 1, linestyle = '', mec= "black")

m = 3

k1_theory, N_theory = largest_deg_dist_p2(N_list,m)
plt.plot(N_theory, k1_theory, linestyle = '--', color = 'gray')

plt.xlabel('N')
plt.ylabel('$k_{1}$')
plt.yscale('log')
plt.xscale('log')

















