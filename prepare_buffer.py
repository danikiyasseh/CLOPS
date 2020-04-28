#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:27:21 2020

@author: Dani Kiyasseh
"""

import numpy as np
import random
from prepare_miscellaneous import obtain_loss_function

#%%
""" Functions in this script:
    1) obtain_indices_for_buffer
    2) obtain_random_storage_indices
    3) obtain_random_retrieval_buffer_dict
    4) obtain_criterion
"""

#%%

def obtain_indices_for_buffer(current_task_index,tracked_loss,tracked_instance_params_dict,current_name,storage_percent,highest=True):#,trial):
    """ Function to Obtain Indices to Write Into Buffer """
    #torch.save(tracked_loss,'/home/scro3517/Desktop/tracked_loss')
    print(current_name)
    if highest == True:
        reverse = True
    elif highest == False:
        reverse = False
    
    tracked_instance_params_list = tracked_instance_params_dict[current_name]
    #print(tracked_instance_params_list)
    
    aul_dict = dict()
#    if trial == 'ER-MIR':
#        for index,loss_over_time in tracked_loss.items():
#            final_loss = np.min(loss_over_time)
#            aul_dict[index] = final_loss
#    else:
    #for index,loss_list in tracked_loss.items():
    for index,param_over_time in tracked_instance_params_list.items():
        #mean_loss = np.trapz(loss_list)
        #print(param_over_time)
        mean_alpha = np.trapz(param_over_time)
        aul_dict[index] = mean_alpha
    
    #print(aul_dict)
    sorted_aul_dict = dict(sorted(aul_dict.items(),key=lambda x:x[1],reverse=reverse))
    tot_samples = len(sorted_aul_dict)
    fraction_to_place_into_buffer = storage_percent[current_task_index] #0.1 #10% of labelled training data from previous task
    nsamples = int(tot_samples*fraction_to_place_into_buffer)
    buffered_indices = list(sorted_aul_dict.keys())[:nsamples] #top-k samples 
    #print(buffered_indices)
    return buffered_indices

def obtain_random_storage_indices(current_task_index,storage_percent,nsamples_in_current_task):
    """ Obtain Random Indices to Write into Buffer """
    fraction_to_place_into_buffer = storage_percent[current_task_index]
    nsamples = int(nsamples_in_current_task*fraction_to_place_into_buffer)
    buffered_indices = random.sample(list(np.arange(nsamples_in_current_task)),nsamples)
    ### challenge is that current task is augmented - we need to find originall nsamples ###
    print('Random Storage Indices Obtained!')
    return buffered_indices
    
def obtain_random_retrieval_buffer_dict(storage_buffer_dict,acquisition_percent):
    """ Obtain Random Indices From Each Task in Buffer to Replay """
    retrieval_buffer_dict = dict()
    for t,(task_name,task_indices) in enumerate(storage_buffer_dict.items()):
        task_nsamples = len(task_indices)
        fraction_to_acquire_from_buffer = acquisition_percent[t]
        nsamples = int(task_nsamples*fraction_to_acquire_from_buffer)
        task_indices = random.sample(list(np.arange(task_nsamples)),nsamples)
        retrieval_buffer_dict[task_name] = task_indices
    print('Random Retrieval Indices Obtained!')
    return retrieval_buffer_dict

def obtain_criterion(phase,models_list,classification,dataloaders_list,pos_weight=1,imbalance_penalty=None):
    if 'train' in phase:
        [model.train() for model in models_list]
        per_sample_loss_dict, criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list,pos_weight,imbalance_penalty)
    elif phase == 'val' or phase == 'test':
        [model.eval() for model in models_list]
        criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list,pos_weight,imbalance_penalty)
    
    return criterion,criterion_single

