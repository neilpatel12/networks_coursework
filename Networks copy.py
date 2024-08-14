#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:42:54 2022

@author: neilpatel

A class to generate degree distributions and largest degree distributions
 for Phase1, Phase2 and Phase3 growing networks
"""
import numpy as np
import numpy.random as rnd
from random import sample

class Phase:
    
    '''
    phase = int, 1,2,or 3, if None then raise error
    
    k1_dist = bool, if False then no k1_dist generated, as this may be much faster
    
    N_max = max system size
    '''
    def __init__(self, phase = None, m = None, N_max = None, ldd = [], dd = True):
        if m == None or N_max == None:
            print('Enter m and N_max')
        self.m = m
        self.N_max = N_max
        self.ldd = ldd
        self.dd = dd
        N_init = m

        self.k1_meas_list = []
        self.N_meas_list = []
        self.N_added_list = np.arange(N_init + 1, N_max + 1) # excludes initial graph nodes
        self.N_tot_list = np.arange(1, N_max + 1) # includes initial graph nodes
        #gen complete graph with N_init = m for initial graph
        if phase == None:
            print('put in phase = 1 or 2 or 3') 
            
        if phase ==1 or phase ==2:   
            self.attachment_list = []
            for i in range(1, N_init + 1):
                self.attachment_list += np.full(N_init - 1, i).tolist()
        elif phase == 3:
             assert m % 2 == 0, 'm must be even for phase 3'
             self.N_degree = [N_init/2]*N_init
             self.attachment_list = np.array(range(1, N_init + 1))
             self.attachment_list = list(np.repeat(self.attachment_list, N_init - 1))
             self.attachment_list = []
             for i in range(1, N_init + 1):
                 self.attachment_list += np.full(N_init - 1, i).tolist()
            
            

        if phase == 1:
            print('Phase 1')
            for N in range(N_init + 1, N_max + 1):
                #add node with m edges with preferential attachment while N< N_max to get final sys size N_max
                chosen_list = []
                self.vertex_set = np.arange(1, N).tolist()
                possible_choices_list = (self.attachment_list).copy() #key!
                while len(chosen_list) < m:
                    #choose random element from attachment list, if its already been chosen, pick again
                    ###### this is the key part, which list you choose from!
                    rnd_index = rnd.randint(0, len(possible_choices_list))
                    choice = possible_choices_list[rnd_index]
                    
                    if not (choice in chosen_list):
                        chosen_list.append(choice)
                        
                    #possible_choices_list[:] = [x for x in possible_choices_list if x != choice]
                    #print('possible_choices_list', possible_choices_list)
                    
                        
                new_node_with_edges = np.full(m, N).tolist()
                
                self.attachment_list += new_node_with_edges + chosen_list
                
                if N in self.ldd:
                	# find k1
                    k1 = max(self.get_k_list())
                    (self.k1_meas_list).append(k1)
                    (self.N_meas_list).append(self.N_count + 1)

            self.get_attributes()

        elif phase == 2:
            print('Phase 2')
            #add node with m edges with random attachment while N< N_max to get final sys size N_max

            for N in range(N_init + 1, N_max + 1):
                #add node with m edges with preferential attachment while N< N_max to get final sys size N_max
                chosen_list = []
                #vertex_set = list(set(self.attachment_list)) # alternatively np.arange(1, N).tolist()
                # if in need of faster method then set
                self.vertex_set = np.arange(1, N).tolist() # this is quicker, but above is clearer
                #print('vert set check', vertex_set == vert2)
                possible_choices_list = (self.vertex_set).copy() # key!
                while len(chosen_list) < m:
                    #choose random element from attachment list, if its already been chosen, pick again
                    ###### this is the key part, which list you choose from!
                    rnd_index = rnd.randint(0, len(possible_choices_list))
                    choice = possible_choices_list[rnd_index]
                    if not(choice in chosen_list):
                        chosen_list.append(choice)
                    #possible_choices_list[:] = [x for x in possible_choices_list if x != choice]
                    #print('possible_choices_list', possible_choices_list)
                    
                        
                new_node_with_edges = np.full(m, N).tolist()
                
                self.attachment_list += new_node_with_edges + chosen_list
   
                if N in self.ldd:
                	# find k1
                    k1 = max(self.get_k_list())
                    (self.k1_meas_list).append(k1)
                    (self.N_meas_list).append(self.N_count + 1)

      
            self.get_attributes()

        elif phase == 3:
            print('Phase 3 TBC')
            
            for N in range(N_init + 1, N_max + 1):
                #add node with m edges with preferential attachment while N< N_max to get final sys size N_max
                chosen_list_RA = []
                self.vertex_set = np.arange(1, N).tolist() # this is quicker, but above is clearer

                possible_choices_list = (self.vertex_set).copy() # key!
                while len(chosen_list) < int(m/2):
                    rnd_index = rnd.randint(0, len(possible_choices_list))
                    choice = possible_choices_list[rnd_index]
                    if not (choice in chosen_list_RA):
                        chosen_list_RA.append(choice)
                    
                    
                    #possible_choices_list[:] = [x for x in possible_choices_list if x != choice]
                    #print('possible_choices_list', possible_choices_list)
                #pick pa pair
                chosen_list_PA = []
                self.vertex_set = np.arange(1, N).tolist()
                possible_choices_list = (self.attachment_list).copy() #key!
                while len(chosen_list) < (m/2):
                    #choose random element from attachment list, if its already been chosen, pick again
                    ###### this is the key part, which list you choose from!
                    rnd_index1 = rnd.randint(0, len(possible_choices_list))
                    rnd_index2 = rnd.randint(0, len(possible_choices_list))
                    choice1 = possible_choices_list[rnd_index1]
                    choice2 = possible_choices_list[rnd_index2]
                    choices12 = [choice1, choice2]
                    chosen_list_PA.extend(choices12)
                    #possible_choices_list[:] = [x for x in possible_choices_list if x != choice]
                    #print('possible_choices_list', possible_choices_list)
                    
                
                        
                new_node_with_edges = np.full(m, N).tolist()
                self.attachment_list += new_node_with_edges + chosen_list_RA + chosen_list_PA
            self.get_attributes()
            
            
                  
        else:
            print('put in valid phase') 
            
            
            
    ###    functions   
    
    def get_attributes(self):
        self.k_list = self.get_k_list()
        if self.dd:
            self.degree_dist = self.get_degree_dist()
        #self.normalisation_check()
        if len(self.ldd) > 0:
            self.k1_dist = self.get_k1_dist()

        
    def get_k_list(self):
        k_list = []
        #vertex_set = list(set(self.attachment_list))
        vertex_set = self.vertex_set
        k_list = []
        for vertex in vertex_set:
            k = (self.attachment_list).count(vertex) #degree of a given vertex
            k_list.append(k)
            assert k>=self.m, f'minimum degree must be m i.e k>=m, here m = {self.m} and k = {k}'

        return k_list
    
    def get_degree_dist(self): #number of nodes with degree k
        '''
        deg_dist = trial.degree_dist
        p_k = deg_dist[:,1]
        k_set = deg_dist[:,0]
        '''
        k_set = list(set(self.k_list))
        n_list = []
        for k in k_set:
            n = (self.k_list).count(k) #number of times a given degree appears
            n_list.append(n)
            
        prob_list = np.array(n_list)/len(self.N_tot_list)    # whats the normalisation fir the probability
        deg_dist = np.column_stack((k_set, prob_list))

        return deg_dist
    
    def get_k1_dist(self):
        '''
        k1_dist = trial.k1_dist
        N_list = k1_dist[:,1]
        k1_list = k1_dist[:,0]
        '''
        k1_dist = np.column_stack((self.k1_list, self.N_added_list))
        return k1_dist
    
    def normalisation_check(self):
        norm = sum(self.degree_dist[:,1])
        print(f'Normalisation Check: The sum of pk for all k is {norm:.03f}, it should be 1')
            
   