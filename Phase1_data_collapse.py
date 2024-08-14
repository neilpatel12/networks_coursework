#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:51:12 2022

@author: neilpatel
"""
import numpy as np
import matplotlib.pyplot as plt
from Provided_code.logbin_2020 import logbin

m = 3 #[3 ** n for n in [0,1,2,3,4,5,6]]
no_trials_list = [100000, 10000, 1000, 100, 10] # [10 ** n for n in [4,3,3,3,2,2]]
N_max_list = [100,1000,10000,100000,1000000]

color_list = np.array(['red', 'orange', 'yellow', 'lightgreen', 'green', 'lightblue', 'blue', 'indigo'])
marker_list = [".", "^", "s", "*", "x"]

def theoretical_deg_dist_p1(m, k_list):
    #k_array = np.linspace(min(k_list), max(k_list), 100)
    k_array = np.array(k_list)
    p_k = 2 * m * (m +1)/ ((k_array)*(k_array+1)*(k_array+2))
    return p_k, k_array

def largest_deg_dist_p1(N,m):
    k1 = 0.5 * (-1 + np.sqrt(1 + 4 * N * m * (m+1)))
    return k1


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
print(f'm = {m}')
#%%
N_max = 100
no_trials = 100000
deg_dist2 = np.load(f'Code/Data/dc/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)
k_list2 = np.load(f'Code/Data/dc/Phase1_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)


N_max = 1000
no_trials = 10000
deg_dist3 = np.load(f'Code/Data/dc/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)
k_list3 = np.load(f'Code/Data/dc/Phase1_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)


N_max = 10000
no_trials = 1000
deg_dist4 = np.load(f'Code/Data/dc/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)
k_list4 = np.load(f'Code/Data/dc/Phase1_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)


N_max = 100000
no_trials = 100
deg_dist5 = np.load(f'Code/Data/dc/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)
k_list5 = np.load(f'Code/Data/dc/Phase1_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)


N_max = 1000000
no_trials = 10
deg_dist6 = np.load(f'Code/Data/dc/Phase1_deg_dist/deg_dist_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)
k_list6 = np.load(f'Code/Data/dc/Phase1_k_list/k_list_m_{m}_N_max_{N_max}_no_trials_{no_trials}.npy', allow_pickle = True)



deg_dist_list = [deg_dist2, deg_dist3, deg_dist4, deg_dist5, deg_dist6]
k_list_list   = [k_list2, k_list3, k_list4, k_list5, k_list6]

print('data loaded')

#%%
scale = 1.1
k_bins2, p_k_binned2 = logbin(k_list2.flatten(), scale)
k_bins3, p_k_binned3 = logbin(k_list3.flatten(), scale)
k_bins4, p_k_binned4 = logbin(k_list4.flatten(), scale)
k_bins5, p_k_binned5 = logbin(k_list5.flatten(), scale)
k_bins6, p_k_binned6 = logbin(k_list6.flatten(), scale)

k_bins_list = [k_bins2, k_bins3, k_bins4, k_bins5, k_bins6]
p_k_binned_list = [p_k_binned2, p_k_binned3, p_k_binned4, p_k_binned5, p_k_binned6]
print('log binned')
#%%

deg_dist2_long = np.concatenate(([deg_dist2[i] for i in range(len(deg_dist2))]))
deg_dist3_long = np.concatenate(([deg_dist3[i] for i in range(len(deg_dist3))]))
deg_dist4_long = np.concatenate(([deg_dist4[i] for i in range(len(deg_dist4))]))
deg_dist5_long = np.concatenate(([deg_dist5[i] for i in range(len(deg_dist5))]))
deg_dist6_long = np.concatenate(([deg_dist6[i] for i in range(len(deg_dist6))]))


k_list2_long = np.concatenate(([k_list2[i] for i in range(len(k_list2))]))
k_list3_long = np.concatenate(([k_list3[i] for i in range(len(k_list3))]))
k_list4_long = np.concatenate(([k_list4[i] for i in range(len(k_list4))]))
k_list5_long = np.concatenate(([k_list5[i] for i in range(len(k_list5))]))
k_list6_long = np.concatenate(([k_list6[i] for i in range(len(k_list6))]))

#stats for each m value

deg_dist_long_list = [deg_dist2_long, deg_dist3_long, deg_dist4_long, deg_dist5_long, deg_dist6_long]
k_list_long_list = [k_list2_long, k_list3_long, k_list4_long, k_list5_long, k_list6_long]

print('long lists done')

#%%
plt.scatter(deg_dist2_long[:,0], deg_dist2_long[:,1])
plt.yscale('log')
plt.xscale('log')

#%% actual plots in report
if False:
    plot_raw = True
    plot_raw_all = True    
    plot_logbinned = False
    plot_theory = True

if 0: # PA_no_data_collapse
    plot_raw = 0
    plot_raw_all = 0
    plot_logbinned = 1
    plot_theory = 1
    vc_only = 0
    tot_dc = 0

if 0: # PA_data_collapse_vc
    plot_raw = 0
    plot_raw_all = 0
    plot_logbinned = 0
    plot_theory = 0
    vc_only = 1
    tot_dc = 0
    
if 1: # PA_data_collapse_tot_dc
    plot_raw = 0
    plot_raw_all = 0
    plot_logbinned = 0
    plot_theory = 0
    vc_only = 0
    tot_dc = 1
    
    
    
plt.figure()
idx_trial = 0

# grad_list = []

for i in range(len(N_max_list) - 1):
    color = color_list[i]
    marker = '.'
    N_max = N_max_list[i]
    if plot_raw:
        deg_dist = deg_dist_list[i][idx_trial]
        p_k = deg_dist[:,1]
        k_set = deg_dist[:,0]
        plt.scatter(k_set, p_k, color = color, label = f'm = {m}', marker = marker)
        plt.xlabel('k')
        plt.ylabel('p(k)')
        
    if plot_raw_all:
        deg_dist_long = deg_dist_long_list[i]
        p_k_long = deg_dist_long[:,1]
        k_set_long = deg_dist_long[:,0]
        plt.scatter(k_set_long, p_k_long, color = color, label = f'm = {m}', marker = marker)
        plt.xlabel('k')
        plt.ylabel('p(k)')
        
        
        
    if plot_logbinned:
        k_bins = k_bins_list[i]
        p_k_binned = p_k_binned_list[i]
        plt.scatter(k_bins, p_k_binned, color = color, label = f'$N$ = {N_max}', edgecolors= "black")
        plt.xlabel('$\\tilde{k}$')
        plt.ylabel('$p_{\\tilde{k}}$')
        #plt.ylabel('p($\\tilde{k}$)')
        
        # k_bins_crop = []
        # p_k_binned_crop = []
        # for j in range(len(k_bins)):
        #     if k_bins[j] < lin_region_ub[i]:
        #         k_bins_crop.append(k_bins[j])
        #         p_k_binned_crop.append(p_k_binned[j])                
        # p,V = np.polyfit(np.log(k_bins_crop), np.log(p_k_binned_crop), 1, cov = True)
        # plt.plot(k_bins_crop,(k_bins_crop ** p[0] * np.e**p[1]), color = 'blue')
        # grad_list.append(p[0])
        # print(f'The gradient is {p[0]} ± {V[0,0]}')
        
    if vc_only:
        k_bins = k_bins_list[i]
        p_k_binned = p_k_binned_list[i]
        p_k_theory, k_theory = theoretical_deg_dist_p1(m, k_bins)
        p_k_binned_vc = np.array(p_k_binned)/ p_k_theory
        plt.scatter(k_bins, p_k_binned_vc, color = color, label = f'$N$ = {N_max}', edgecolors= "black")
        plt.xlabel('$\\tilde{k}$')
        plt.ylabel('$\\frac{p_{\\tilde{k}}}{p_{k}(\infty)}$')
        
    if tot_dc:
        k_bins = k_bins_list[i]
        p_k_binned = p_k_binned_list[i]
        p_k_theory, k_theory = theoretical_deg_dist_p1(m, k_bins)
        p_k_binned_vc = np.array(p_k_binned)/ p_k_theory
        k_bins_hc = k_bins/ largest_deg_dist_p1(N_max, m)
        plt.scatter(k_bins_hc, p_k_binned_vc, color = color, label = f'$N$ = {N_max}', edgecolors= "black")
        plt.xlabel('$\\frac{\\tilde{k}}{k_{1}}$')
        plt.ylabel('$\\frac{p_{\\tilde{k}}}{p_{k}(\infty)}$')
        plt.tight_layout()
        
    if plot_theory:
        k_bins = k_bins_list[i]
        p_k_theory, k_theory = theoretical_deg_dist_p1(m, k_bins)
        plt.plot(k_theory, p_k_theory, color = 'gray', linestyle = '--', zorder = 0)   
        

    #     chisquare = sp.chisquare(p_k)
    
    #KS and chisquare test
    #k_list_long = k_list_long_list[i]
    
plt.tight_layout()
plt.yscale('log')
plt.xscale('log')
plt.legend()







