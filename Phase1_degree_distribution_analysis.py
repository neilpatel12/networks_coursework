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

def theoretical_deg_dist_p1(m, k_list):
    k_array = np.linspace(min(k_list), max(k_list), 100)
    p_k = 2 * m * (m+1) / ((k_array)*(k_array+1)*(k_array+2))
    return p_k, k_array



m_list = [1, 3, 9, 27, 81, 243] #[3 ** n for n in [0,1,2,3,4,5,6]]
no_trials_list = [10000, 1000, 1000, 1000, 100, 100] # [10 ** n for n in [4,3,3,3,2,2]]
N_max = 10000
lin_region_ub = [80, 90, 150, 310, 760, 1300 ]# TBC linear region upper bound degree, measure by eye

color_list = np.array(['red', 'orange', 'yellow', 'lightgreen', 'green', 'lightblue', 'blue', 'indigo'])
marker_list = [".", "^", "s", "*", "x"]

    
file_path = '/Users/neilpatel/OneDrive - Imperial College London/Year 3/CandN/Networks Project/'   
    
print('load in data')

#%%
deg_dist1 = np.load(file_path + 'Code/Data/dd/Phase1_deg_dist/deg_dist_m_1_N_max_10000_no_trials_10000.npy', allow_pickle = True)
k_list1 = np.load(file_path + 'Code/Data/dd/Phase1_k_list/k_list_m_1_N_max_10000_no_trials_10000.npy', allow_pickle = True)
    
deg_dist3 = np.load(file_path + 'Code/Data/dd/Phase1_deg_dist/deg_dist_m_3_N_max_10000_no_trials_1000.npy', allow_pickle = True)
k_list3 = np.load(file_path + 'Code/Data/dd/Phase1_k_list/k_list_m_3_N_max_10000_no_trials_1000.npy', allow_pickle = True)
    
deg_dist9 = np.load(file_path + 'Code/Data/dd/Phase1_deg_dist/deg_dist_m_9_N_max_10000_no_trials_1000.npy', allow_pickle = True)
k_list9 = np.load(file_path + 'Code/Data/dd/Phase1_k_list/k_list_m_9_N_max_10000_no_trials_1000.npy', allow_pickle = True)
    
deg_dist27 = np.load(file_path + 'Code/Data/dd/Phase1_deg_dist/deg_dist_m_27_N_max_10000_no_trials_1000.npy', allow_pickle = True)
k_list27 = np.load(file_path + 'Code/Data/dd/Phase1_k_list/k_list_m_27_N_max_10000_no_trials_1000.npy', allow_pickle = True)
    
deg_dist81 = np.load(file_path + 'Code/Data/dd/Phase1_deg_dist/deg_dist_m_81_N_max_10000_no_trials_100.npy', allow_pickle = True)
k_list81 = np.load(file_path + 'Code/Data/dd/Phase1_k_list/k_list_m_81_N_max_10000_no_trials_100.npy', allow_pickle = True)

deg_dist243 = np.load(file_path + 'Code/Data/dd/Phase1_deg_dist/deg_dist_m_243_N_max_10000_no_trials_100.npy', allow_pickle = True)
k_list243 = np.load(file_path + 'Code/Data/dd/Phase1_k_list/k_list_m_243_N_max_10000_no_trials_100.npy', allow_pickle = True)
    
deg_dist_list = [deg_dist1, deg_dist3, deg_dist9, deg_dist27, deg_dist81, deg_dist243]
k_list_list   = [k_list1, k_list3, k_list9, k_list27, k_list81, k_list243]

print('data loaded')

#%%
scale = 1.05
k_bins1, p_k_binned1 = logbin(k_list1.flatten(), scale)
k_bins3, p_k_binned3 = logbin(k_list3.flatten(), scale)
k_bins9, p_k_binned9 = logbin(k_list9.flatten(), scale)
k_bins27, p_k_binned27 = logbin(k_list27.flatten(), scale)
k_bins81, p_k_binned81 = logbin(k_list81.flatten(), scale)
k_bins243, p_k_binned243 = logbin(k_list243.flatten(), scale)

k_bins_list = [k_bins1, k_bins3, k_bins9, k_bins27, k_bins81, k_bins243]
p_k_binned_list = [p_k_binned1, p_k_binned3, p_k_binned9, p_k_binned27, p_k_binned81, p_k_binned243]
print('log binned')
#%%

deg_dist1_long = np.concatenate(([deg_dist1[i] for i in range(len(deg_dist1))]))
deg_dist3_long = np.concatenate(([deg_dist3[i] for i in range(len(deg_dist3))]))
deg_dist9_long = np.concatenate(([deg_dist9[i] for i in range(len(deg_dist9))]))
deg_dist27_long = np.concatenate(([deg_dist27[i] for i in range(len(deg_dist27))]))
deg_dist81_long = np.concatenate(([deg_dist81[i] for i in range(len(deg_dist81))]))
deg_dist243_long = np.concatenate(([deg_dist243[i] for i in range(len(deg_dist243))]))

k_list1_long = np.concatenate(([k_list1[i] for i in range(len(k_list1))]))
k_list3_long = np.concatenate(([k_list3[i] for i in range(len(k_list3))]))
k_list9_long = np.concatenate(([k_list9[i] for i in range(len(k_list9))]))
k_list27_long = np.concatenate(([k_list27[i] for i in range(len(k_list27))]))
k_list81_long = np.concatenate(([k_list81[i] for i in range(len(k_list81))]))
k_list243_long = np.concatenate(([k_list243[i] for i in range(len(k_list243))]))

#stats for each m value

deg_dist_long_list = [deg_dist1_long, deg_dist3_long, deg_dist9_long, deg_dist27_long, deg_dist81_long, deg_dist243_long]
k_list_long_list = [k_list1_long, k_list3_long, k_list9_long, k_list27_long, k_list81_long, k_list243_long]

plt.scatter(deg_dist1_long[:,0], deg_dist1_long[:,1])
plt.yscale('log')
plt.xscale('log')

#%% actual plots in report

def theoretical_deg_dist_p1_calc(m, k):
    p_k = 2 * m * (m+1) / ((k)*(k+1)*(k+2))
    return p_k

if 1: #PA_deg_dist_and_theory
    plot_raw = 0
    plot_raw_all = 1
    plot_logbinned = 0
    plot_theory = 1
    plot_grad = 0
    
if 0: #PA_logbinned_and_theory
    plot_raw = 0
    plot_raw_all = 0
    plot_logbinned = 1
    plot_theory = 1
    plot_grad = 0
plt.figure()
idx_trial = 0

grad_list = []
chisq_raw_all_list = [] # [0.042945041334077845,
 # 0.04875690297064869,
 # 0.07102044446519139,
 # 0.11824708014839391,
 # 0.21328983251188036,
 # 0.4570373315718694]
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
        
        k_bins_crop = []
        p_k_binned_crop = []
        for j in range(len(k_bins)):
            if k_bins[j] < lin_region_ub[i]:
                k_bins_crop.append(k_bins[j])
                p_k_binned_crop.append(p_k_binned[j])                
        p,V = np.polyfit(np.log(k_bins_crop), np.log(p_k_binned_crop), 1, cov = True)
        if plot_grad:
            plt.plot(k_bins_crop,(k_bins_crop ** p[0] * np.e**p[1]), color = 'blue')
        grad_list.append(p[0])
        print(f'The gradient is {p[0]} ± {V[0,0]}')
        
        
    if plot_theory:
        k_bins = k_bins_list[i]
        p_k_theory, k_theory = theoretical_deg_dist_p1(m, k_bins)
        plt.plot(k_theory, p_k_theory, color = color, linestyle = '--', zorder = 0)
       # chisquare = sp.chisquare(p_k_binned, theoretical_deg_dist_p1_calc(k_bins))
       
        if plot_raw_all:
            p_k_for_chi = p_k_long
            k_for_chi = k_set_long
            p_k_theory_for_chi = theoretical_deg_dist_p1_calc(m, k_for_chi)
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
            p_k_theory_for_chi = theoretical_deg_dist_p1_calc(m, k_for_chi)
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
plot_grad = False
# do for gradient if time
enr_array = np.zeros(len(m_list)) # take mean and std stats
enr_list_list = []
grad_list_list = []
for mi in range(len(m_list)):
    m = m_list[mi]
    no_trials = no_trials_list[mi]
    enr_list = []
    grad_list = []
    for ti in tqdm(range(no_trials)): # i.e. for each trial
        #binned
        color = color_list[mi]
        k_list = k_list_list[mi][ti]
        k_bins, p_k_binned = logbin(k_list, scale = 1.05)
        plt.scatter(k_bins, p_k_binned, color = color, label = f'm = {m}', edgecolors= "black")
        plt.xlabel('$\\tilde{k}$')
        plt.ylabel('$p_{\\tilde{k}}$')
        #plt.ylabel('p($\\tilde{k}$)')
        #plt.ylabel('p($\\tilde{k}$)')
        
        k_bins_crop = []
        p_k_binned_crop = []
        for j in range(len(k_bins)):
            if k_bins[j] < lin_region_ub[mi]:
                k_bins_crop.append(k_bins[j])
                p_k_binned_crop.append(p_k_binned[j])                
        p,V = np.polyfit(np.log(k_bins_crop), np.log(p_k_binned_crop), 1, cov = True)
        if plot_grad:
            plt.plot(k_bins_crop,(k_bins_crop ** p[0] * np.e**p[1]), color = 'blue')
        grad_list.append(p[0])
        #print(f'The gradient is {p[0]} ± {V[0,0]}')
        
        #raw
        # deg_dist = deg_dist_list[mi][ti]
        # p_k = deg_dist[:,1]
        # k_set = deg_dist[:,0]
        

        
        # print E/N ratio, this should be = m. This counts as a check for working code
        edge_node_ratio = 0.5 * sum(k_list)/N_max
        #print(edge_node_ratio)
        enr_list.append(edge_node_ratio)
    enr_list_list.append(np.array(enr_list))
    grad_list_list.append(np.array(grad_list))
    #enr_array[mi] = np.array(enr_list)
    print(f'E({N_max})/N({N_max})  = {edge_node_ratio}, it should be approx {m}')
#%%  
enr_mean = [np.mean(enr_list) for enr_list in enr_list_list]
enr_std = [np.std(enr_list)/np.sqrt(len(enr_list)) for enr_list in enr_list_list]

print('enr_mean', enr_mean)
print('enr_std', enr_std)

grad_mean = [np.mean(grad_list) for grad_list in grad_list_list]
grad_std = [np.std(grad_list)/np.sqrt(len(grad_list)) for grad_list in grad_list_list]

print('grad_mean', grad_mean)
print('grad_std', grad_std)


#enr_mean [0.9999000000000002, 2.999400000000001, 8.995499999999996, 26.962199999999996, 80.66790000000003, 240.0354000000001]
#enr_std [2.220446049250313e-18, 2.8086667748613606e-17, 1.1234667099445443e-16, 1.1234667099445443e-16, 2.842170943040401e-15, 8.526512829121202e-15]


#%%    
    
# data = k_list1[0]

# # sort the data:
# data_sorted = np.sort(data)

# # calculate the proportional values of samples
# p = 1. * np.arange(len(data)) / (len(data) - 1)

# # plot the sorted data:
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.plot(p, data_sorted)
# ax1.set_xlabel('$p$')
# ax1.set_ylabel('$x$')

# ax2 = fig.add_subplot(122)
# ax2.plot(data_sorted, p)
# ax2.set_xlabel('$x$')
# ax2.set_ylabel('$p$')  

# deg_dist = deg_dist1[0]
# k_set = deg_dist[:,0]
# p_k = deg_dist[:,1]

# # ax3 = fig.add_subplot(221)
# # ax3.plot(k_set, p_k)
# # ax3.set_xlabel('$k$')
# # ax3.set_ylabel('$p$')  


    
#%%
# deg_dist = deg_dist1[0]
# k_set = deg_dist[:,0]
# p_k = deg_dist[:,1]
# k_list = k_list1[0]



#%% def cdf

# def cdf(deg_dist, m):
#     k_set = deg_dist[:,0]
#     p_k = deg_dist[:,1]
#     k_set_sorted = sorted(k_set)
#     p_k_sorted = []
#     p_k_sorted_theory = []
#     for i in range(len(k_set_sorted)):
#         k = k_set_sorted[i]
#         idx = np.argwhere(k_set == k)
#         p_k_sorted.append(p_k[idx])
#         p_k_sorted_theory.append(theoretical_deg_dist_p1_calc(m,k))
        
        
#     k_poss = np.arange(min(k_set), max(k_set)+1)
#     k_poss = np.arange(1, 10000)
#     c_k_list = []
#     c_k_theory_list = []
#     for i in range(len(k_poss)):
#         k = k_poss[i]
#         c_k_theory = float(sum(p_k_sorted_theory[:i]))
#         c_k_theory_list.append(c_k_theory)
#         if k in k_set:  
#             c_k = float(sum(p_k_sorted[:i]))
#             c_k_list.append(c_k)
#         else:
#             c_k_list.append(c_k)
            
#     KS_idx = np.argmax(np.abs(np.array(c_k_theory_list) - np.array(c_k_list)))        
#     KS = max(np.abs(np.array(c_k_theory_list) - np.array(c_k_list)))
#     diff_list = np.abs(np.array(c_k_theory_list) - np.array(c_k_list))       
#     return np.array(k_poss), np.array(c_k_list), np.array(c_k_theory_list), diff_list, KS, KS_idx

# def theoretical_deg_dist_p1_calc(m, k):
#     p_k = 2 * m * (m+1) / ((k)*(k+1)*(k+2))
#     return p_k

# m = 1
# k_poss, c_k_list, c_k_theory_list, diff_list, KS, KS_idx = cdf(deg_dist, m)

#%%
import scipy.stats as stats
from numba import njit
#a = stats.kstest(k_list, c_k_theory_list, alternative='two-sided', mode='auto')

def theoretical_deg_dist_p1_calc(m, k):
    p_k = 2 * m * (m+1) / ((k)*(k+1)*(k+2))
    return p_k

# def get_freq_array(k_list, m, N_max): #number of nodes with degree k
#     '''
#     deg_dist = trial.get_degree_dist()
#     p_k = deg_dist[:,1]
#     k_set = deg_dist[:,0]
#     '''
#     k_list = k_list.tolist()
#     k_set = list(set(k_list))
#     k_poss = np.arange(min(k_set), max(k_set)+1)
#     n_list = []
#     for k in k_set:
#         n = (k_list).count(k) #number of times a given degree appears
#         n_list.append(n)
        
#     # fills in zeros for the k values with zero frequency
#     freq_array = np.zeros(len(k_poss)).tolist()
#     freq_array_theory = []
    
    
#     for i in range(len(k_poss)):
#         k = k_poss[i]
#         p_k_theory = theoretical_deg_dist_p1_calc(m,k)
#         n_theory = round(N_max * p_k_theory)
#         freq_array_theory.append(n_theory)
#         if k in k_set:
            
#             idx = int(np.argwhere(k_set == k))
   
#             n = n_list[idx]
#             freq_array[i] = n
#         else:
#             freq_array[i] = 0            
    
#     return freq_array, freq_array_theory

# 

#@njit
def get_n_lists(k_list, m, N_max): #number of nodes with degree k
    '''
    deg_dist = trial.get_degree_dist()
    p_k = deg_dist[:,1]
    k_set = deg_dist[:,0]
    '''
    k_list = k_list.tolist()
    k_set = list(set(k_list))
    n_list = []
    n_list_theory = N_max * theoretical_deg_dist_p1_calc(m,np.array(k_set))
    for k in tqdm(k_set):
        n = (k_list).count(k) #number of times a given degree appears
        n_list.append(n)
    return n_list, n_list_theory

@njit
def get_n_list_numba(k_list, m, N_max): #number of nodes with degree k
    '''
    deg_dist = trial.get_degree_dist()
    p_k = deg_dist[:,1]
    k_set = deg_dist[:,0]
    '''
    k_list = np.array(k_list)
    k_set = set(k_list)
    n_list = []
    #n_list_theory = N_max * theoretical_deg_dist_p1_calc(m,np.array(k_set))
    for k in tqdm(k_set):
        n = np.count_nonzero(k_list == k) #(k_list).count(k) #number of times a given degree appears
        n_list.append(n)
    return n_list
#%%
mi = 0
m = m_list[mi]
k_list_long = k_list_long_list[mi]
freq_array = get_n_list_numba(k_list_long, m, N_max)

#%%
mi = 0
m = m_list[mi]
k_list_long = k_list_long_list[mi]
freq_array, freq_array_theory = get_n_lists(k_list_long, m, N_max)

#%%
N_max = 10000
mi = 0
ti = 0
for mi in range(len(m_list)):
    m = m_list[mi]
    print(f'm = {m}')
    k_list_long = k_list_long_list[mi]
    freq_array, freq_array_theory = get_n_lists(k_list_long, m, N_max)
    chisq, p_val = stats.chisquare(freq_array, freq_array_theory)
    print(f'for m = {m}, with N_max = {N_max}, the chisq = {chisq} with p_val = {p_val}')
    if p_val < 0.05:
        print('The null hypothesis is rejected at the alpha = 0.05 significance level. Boo!')
    elif p_val> 0.05:
        print('The null hypothesis cannot be rejected at the alpha = 0.05 significance level. Yay!')



#%%
plt.figure()
N_max = 10000
mi = 0
ti = 0
for mi in range(len(m_list)):
    m = m_list[mi]
    print(f'm = {m}')
    deg_dist = deg_dist_list[mi][ti]
    p_k =  deg_dist[:,1]
    k_set = deg_dist[:,0]
    p_k_theory = theoretical_deg_dist_p1_calc(m, k_set)
    
    plt.plot(k_set, p_k_theory)
    plt.scatter(k_set, p_k, label = f'm = {m}')
    # chisq, p_val = stats.chisquare(p_k, p_k_theory)
    # print(f'for m = {m}, with N_max = {N_max}, the chisq = {chisq} with p_val = {p_val}')
    # if p_val < 0.05:
    #     print('The null hypothesis is rejected at the alpha = 0.05 significance level. Boo!')
    # elif p_val> 0.05:
    #     print('The null hypothesis cannot be rejected at the alpha = 0.05 significance level. Yay!')

plt.legend()
plt.ylabel('p_k')
plt.xlabel('k')
plt.yscale('log')
plt.xscale('log')






#%%
plt.figure()
plt.plot(k_poss, c_k_list, label = 'Empirical distribution')
plt.plot(k_poss, c_k_theory_list, label = 'Theoretical CDF')
plt.xlabel('$k$')
plt.ylabel('$P(K<k)$')
plt.legend()

#def cdf_theory():
    
#%%
plt.figure()
plt.plot(k_poss, diff_list)
plt.plot(np.array(c_k_theory_list) - np.array(c_k_list))
plt.xlabel('$k$')
plt.ylabel('$P(K<k)_{theory} - P(K<k)_{empirical}$')
plt.legend()
    
    
    
    
    
    
    
    
    

    