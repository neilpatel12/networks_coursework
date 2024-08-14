#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:41:36 2022

@author: neilpatel
"""

import numpy as np
from Provided_code.logbin_2020 import logbin
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sp

import scipy.stats as stats

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

def theoretical_deg_dist_p3(m, k_list):
    k_array = np.linspace(min(k_list), max(k_list), 100)
    k = k_array
    p_k = (3 * m * (3 * m + 2))/(2*(k+m)*(k+m+1)*(k+m+2))
    return p_k, k_array

m_list = [4,8,16,32] #[3 ** n for n in [0,1,2,3,4,5,6]]
no_trials_list = [10000, 10000, 1000, 100] # [10 ** n for n in [4,3,3,3,2,2]]
N_max = 10000
#lin_region_ub = [80, 90, 150, 310, 760, 1300 ]# TBC linear region upper bound degree, measure by eye

color_list = np.array(['red', 'orange', 'yellow', 'lightgreen', 'green', 'lightblue', 'blue', 'indigo'])
marker_list = [".", "^", "s", "*", "x"]

    
file_path = '/Users/neilpatel/OneDrive - Imperial College London/Year 3/CandN/Networks Project/'   
    
print('load in data')

#%%
deg_dist4 = np.load(file_path + 'Code/Data/dd/Phase3_deg_dist/deg_dist_m_4_N_max_10000_no_trials_10000.npy', allow_pickle = True)
k_list4 = np.load(file_path + 'Code/Data/dd/Phase3_k_list/k_list_m_4_N_max_10000_no_trials_10000.npy', allow_pickle = True)

deg_dist8 = np.load(file_path + 'Code/Data/dd/Phase3_deg_dist/deg_dist_m_8_N_max_10000_no_trials_10000.npy', allow_pickle = True)
k_list8 = np.load(file_path + 'Code/Data/dd/Phase3_k_list/k_list_m_8_N_max_10000_no_trials_10000.npy', allow_pickle = True)
    
deg_dist16 = np.load(file_path + 'Code/Data/dd/Phase3_deg_dist/deg_dist_m_16_N_max_10000_no_trials_1000.npy', allow_pickle = True)
k_list16 = np.load(file_path + 'Code/Data/dd/Phase3_k_list/k_list_m_16_N_max_10000_no_trials_1000.npy', allow_pickle = True)
    
deg_dist32 = np.load(file_path + 'Code/Data/dd/Phase3_deg_dist/deg_dist_m_32_N_max_10000_no_trials_100.npy', allow_pickle = True)
k_list32 = np.load(file_path + 'Code/Data/dd/Phase3_k_list/k_list_m_32_N_max_10000_no_trials_100.npy', allow_pickle = True)
    
deg_dist_list =[]

print('data loaded')

#%%
scale = 1.05
k_bins4, p_k_binned4 = logbin(k_list4.flatten(), scale)
k_bins8, p_k_binned8 = logbin(k_list8.flatten(), scale)
k_bins16, p_k_binned16 = logbin(k_list16.flatten(), scale)
k_bins32, p_k_binned32 = logbin(k_list32.flatten(), scale)

k_bins_list = [k_bins4, k_bins8, k_bins16, k_bins32]
p_k_binned_list = [p_k_binned4, p_k_binned8, p_k_binned16, p_k_binned32]
print('log binned')
#%%

deg_dist4_long = np.concatenate(([deg_dist4[i] for i in range(len(deg_dist4))]))
deg_dist8_long = np.concatenate(([deg_dist8[i] for i in range(len(deg_dist8))]))
deg_dist16_long = np.concatenate(([deg_dist16[i] for i in range(len(deg_dist16))]))
deg_dist32_long = np.concatenate(([deg_dist32[i] for i in range(len(deg_dist32))]))

k_list4_long = np.concatenate(([k_list4[i] for i in range(len(k_list4))]))
k_list8_long = np.concatenate(([k_list8[i] for i in range(len(k_list8))]))
k_list16_long = np.concatenate(([k_list16[i] for i in range(len(k_list16))]))
k_list32_long = np.concatenate(([k_list32[i] for i in range(len(k_list32))]))

#stats for each m value

deg_dist_long_list = [deg_dist4_long, deg_dist8_long, deg_dist16_long, deg_dist32_long]
k_list_long_list = [k_list4_long, k_list8_long, k_list16_long, k_list32_long]

print('long lists')
#%%
plt.scatter(deg_dist4_long[:,0], deg_dist4_long[:,1])
plt.yscale('log')
plt.xscale('log')

#%% actual plots in report

#m_list = m_list[1:] # for time being until m = 4 loads

def theoretical_deg_dist_p3_calc(m, k):
    p_k = (3 * m * (3 * m + 2))/(2*(k+m)*(k+m+1)*(k+m+2))
    return p_k


if 1: #EV_deg_dist -- NOT IN REPORT
    plot_raw = 0
    plot_raw_all = 1    
    plot_logbinned = 0
    plot_theory = 1
    plot_grad = 0
    
if 0: #EV_deg_dist
    plot_raw = 0
    plot_raw_all = 0
    plot_logbinned = 1
    plot_theory = 1
    plot_grad = 0
plt.figure()
idx_trial = 0

# grad_list = []

# for i in range(len(m_list)):
#     color = color_list[i]
#     marker = '.'
#     m = m_list[i]
#     if plot_raw:
#         deg_dist = deg_dist_list[i][idx_trial]
#         p_k = deg_dist[:,1]
#         k_set = deg_dist[:,0]
#         plt.scatter(k_set, p_k, color = color, label = f'm = {m}', marker = marker, edgecolors= "black")
#         plt.xlabel('$k$')
#         plt.ylabel('$p_{k}$')
#         # plt.xlabel('k')
#         # plt.ylabel('p(k)')
        
#     if plot_raw_all:
#         deg_dist_long = deg_dist_long_list[i]
#         p_k_long = deg_dist_long[:,1]
#         k_set_long = deg_dist_long[:,0]
#         plt.scatter(k_set_long, p_k_long, color = color, label = f'm = {m}', marker = marker, edgecolors= "black")
#         plt.xlabel('$k$')
#         plt.ylabel('$p_{k}$')
#         # plt.xlabel('k')
#         # plt.ylabel('p(k)')
        
#     if plot_logbinned:
#         k_bins = k_bins_list[i]
#         p_k_binned = p_k_binned_list[i]
#         plt.scatter(k_bins, p_k_binned, color = color, label = f'm = {m}', edgecolors= "black")
#         plt.xlabel('$\\tilde{k}$')
#         plt.ylabel('$p_{\\tilde{k}}$')
#         #plt.ylabel('p($\\tilde{k}$)')
#         #plt.ylabel('p($\\tilde{k}$)')
        
#         # k_bins_crop = []
#         # p_k_binned_crop = []
#         # for j in range(len(k_bins)):
#         #     if k_bins[j] < lin_region_ub[i]:
#         #         k_bins_crop.append(k_bins[j])
#         #         p_k_binned_crop.append(p_k_binned[j])                
#         # p,V = np.polyfit(np.log(k_bins_crop), np.log(p_k_binned_crop), 1, cov = True)
#         # if plot_grad:
#         #     plt.plot(k_bins_crop,(k_bins_crop ** p[0] * np.e**p[1]), color = 'blue')
#         # grad_list.append(p[0])
#         # print(f'The gradient is {p[0]} ± {V[0,0]}')
        
        
#     if plot_theory:
#         k_bins = k_bins_list[i]
#         p_k_theory, k_theory = theoretical_deg_dist_p3(m, k_bins)
#         plt.plot(k_theory, p_k_theory, color = color, linestyle = '--', zorder = 0)
#        # chisquare = sp.chisquare(p_k)
    
#     #KS and chisquare test
#     #k_list_long = k_list_long_list[i]
# plt.tight_layout()
# plt.yscale('log')
# plt.xscale('log')
# plt.legend(loc = 'lower left')

grad_list = []
chisq_raw_all_list = [] 
chisq_logbin_list = []

for i in range(len(m_list)):
    color = color_list[i]
    marker = '.'
    m = m_list[i]
    if plot_raw:
        deg_dist = deg_dist_list[i][idx_trial]
        p_k = deg_dist[:,1]
        k_set = deg_dist[:,0]
        plt.scatter(k_set, p_k, color = color, label = f'm = {m}', marker = marker, edgecolors= "black")
        plt.xlabel('$k$')
        plt.ylabel('$p_{k}$')
        # plt.xlabel('k')
        # plt.ylabel('p(k)')
        
    if plot_raw_all:
        deg_dist_long = deg_dist_long_list[i]
        p_k_long = deg_dist_long[:,1]
        k_set_long = deg_dist_long[:,0]
        plt.scatter(k_set_long, p_k_long, color = color, label = f'm = {m}', marker = marker, edgecolors= "black")
        plt.xlabel('$k$')
        plt.ylabel('$p_{k}$')
        # plt.xlabel('k')
        # plt.ylabel('p(k)')
        
    if plot_logbinned:
        k_bins = k_bins_list[i]
        p_k_binned = p_k_binned_list[i]
        plt.scatter(k_bins, p_k_binned, color = color, label = f'm = {m}', edgecolors= "black")
        plt.xlabel('$\\tilde{k}$')
        plt.ylabel('$p_{\\tilde{k}}$')
        #plt.ylabel('p($\\tilde{k}$)')
        #plt.ylabel('p($\\tilde{k}$)')
        
        # k_bins_crop = []
        # p_k_binned_crop = []
        # for j in range(len(k_bins)):
        #     if k_bins[j] < lin_region_ub[i]:
        #         k_bins_crop.append(k_bins[j])
        #         p_k_binned_crop.append(p_k_binned[j])                
        # p,V = np.polyfit(np.log(k_bins_crop), np.log(p_k_binned_crop), 1, cov = True)
        # if plot_grad:
        #     plt.plot(k_bins_crop,(k_bins_crop ** p[0] * np.e**p[1]), color = 'blue')
        # grad_list.append(p[0])
        # print(f'The gradient is {p[0]} ± {V[0,0]}')
        
        
    if plot_theory:
        k_bins = k_bins_list[i]
        p_k_theory, k_theory = theoretical_deg_dist_p3(m, k_bins)
        plt.plot(k_theory, p_k_theory, color = color, linestyle = '--', zorder = 0)
       # chisquare = sp.chisquare(p_k_binned, theoretical_deg_dist_p1_calc(k_bins))
       
        if plot_raw_all:
            p_k_for_chi = p_k_long
            k_for_chi = k_set_long
            p_k_theory_for_chi = theoretical_deg_dist_p3_calc(m, k_for_chi)
            chisq, p_val = stats.chisquare(p_k_for_chi/sum(p_k_for_chi),  p_k_theory_for_chi/sum(p_k_theory_for_chi))
            print(f'for m = {m}, with N_max = {N_max}, the chisq = {chisq} with p_val = {p_val}')
            chisq_raw_all_list.append(chisq)
            if p_val < 0.05:
                print('The null hypothesis is rejected at the alpha = 0.05 significance level. Boo!')
            elif p_val> 0.05:
                print('The null hypothesis cannot be rejected at the alpha = 0.05 significance level. Yay!')

        if plot_logbinned:
            p_k_for_chi = p_k_binned
            k_for_chi = k_bins
            p_k_theory_for_chi = theoretical_deg_dist_p3_calc(m, k_for_chi)
            chisq, p_val = stats.chisquare(p_k_for_chi/sum(p_k_for_chi),  p_k_theory_for_chi/sum(p_k_theory_for_chi))
            print(f'for m = {m}, with N_max = {N_max}, the chisq = {chisq} with p_val = {p_val}')
            chisq_logbin_list.append(chisq)
            if p_val < 0.05:
                print('The null hypothesis is rejected at the alpha = 0.05 significance level. Boo!')
            elif p_val> 0.05:
                print('The null hypothesis cannot be rejected at the alpha = 0.05 significance level. Yay!')

     
    #KS and chisquare test
    #k_list_long = k_list_long_list[i]

plt.tight_layout()
plt.yscale('log')
plt.xscale('log')
plt.legend(loc = 'lower left')

#%% 
mean_grad_all_m = np.mean(grad_list)

range_grad_all_m = np.abs(max(grad_list) - min(grad_list))

print(f'overall grad = {mean_grad_all_m:.03f} ± {range_grad_all_m:.03f}')



#%% 

# do for gradient if time
enr_array = np.zeros(len(m_list)) # take mean and std stats
enr_list_list = []
for mi in range(len(m_list)):
    m = m_list[mi]
    no_trials = no_trials_list[mi]
    enr_list = []
    for ti in tqdm(range(no_trials)): # i.e. for each trial
    
        #raw
        # deg_dist = deg_dist_list[mi][ti]
        # p_k = deg_dist[:,1]
        # k_set = deg_dist[:,0]
        
        #binned
        k_list = k_list_list[mi][ti]
        
        # print E/N ratio, this should be = m. This counts as a check for working code
        edge_node_ratio = 0.5 * sum(k_list)/N_max
        enr_list.append(edge_node_ratio)
    enr_list_list.append(np.array(enr_list))
    #enr_array[mi] = np.array(enr_list)
    print(f'E({N_max})/N({N_max})  = {edge_node_ratio}, it should be approx {m}')
#%%  
enr_mean = [np.mean(enr_list) for enr_list in enr_list_list]
enr_std = [np.std(enr_list)/np.sqrt(len(enr_list)) for enr_list in enr_list_list]

print('enr_mean', enr_mean)
print('enr_std', enr_std)

#enr_mean [0.9999000000000002, 2.999400000000001, 8.995499999999996, 26.962199999999996, 80.66790000000003, 240.0354000000001]
#enr_std [2.220446049250313e-18, 2.8086667748613606e-17, 1.1234667099445443e-16, 1.1234667099445443e-16, 2.842170943040401e-15, 8.526512829121202e-15]


#%%    
    
data = k_list1[0]

# sort the data:
data_sorted = np.sort(data)

# calculate the proportional values of samples
p = 1. * np.arange(len(data)) / (len(data) - 1)

# plot the sorted data:
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(p, data_sorted)
ax1.set_xlabel('$p$')
ax1.set_ylabel('$x$')

ax2 = fig.add_subplot(122)
ax2.plot(data_sorted, p)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$p$')  

deg_dist = deg_dist1[0]
k_set = deg_dist[0]
p_k = deg_dist[1]

# ax3 = fig.add_subplot(221)
# ax3.plot(k_set, p_k)
# ax3.set_xlabel('$k$')
# ax3.set_ylabel('$p$')  


    
#%%
deg_dist = deg_dist1[0]
k_set = deg_dist[0]
p_k = deg_dist[1]
plt.plot()
    
    
    
    
    
    
    
    

    