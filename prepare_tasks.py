#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:24:59 2020

@author: Dani Kiyasseh
"""

import random
import numpy as np

#%%
""" Functions in this script:
    1) modify_dataset_order_for_multi_task_learning
    2) obtain_dataset_order_for_curriculum
    3) obtain_dataset_order
    4) obtain_dicts
"""
#%%

""" When to Transition from One Task to the Next """

""" This is a 1-1 Mapping Between All Possible Datasets and Ideal BS and LR Found for Them """
dataset_list = ['physionet','physionet2017','cardiology','ptb','fetal','physionet2016','physionet2020','chapman','uci_emg']#,'cipa']
batch_size_list = [256, 256, 16, 64, 64, 256, 256, 256, 256]#, 512]
lr_list = [1e-4, 1e-4, 1e-4, 5e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5]#, 1e-4]
data2bs_dict = dict(zip(dataset_list,batch_size_list))
data2lr_dict = dict(zip(dataset_list,lr_list))

def modify_dataset_order_for_multi_task_learning(new_task_datasets,new_task_modalities,new_task_leads,new_task_class_pairs,new_task_fractions):
    new_task_datasets = [new_task_datasets]
    new_task_modalities = [new_task_modalities]
    new_task_leads = [new_task_leads]
    new_task_class_pairs = [new_task_class_pairs]
    new_task_fractions = [new_task_fractions]
    return new_task_datasets,new_task_modalities,new_task_leads,new_task_class_pairs,new_task_fractions

def obtain_dataset_order_for_curriculum(trial,cl_scenario,dataset_name,fraction,order):
    if cl_scenario == 'Class-IL':
        if dataset_name == 'cardiology':
            task_epochs = 20
            ntasks = 6
            new_task_datasets = ['cardiology'] * ntasks
            new_task_modalities = [['ecg']] * ntasks
            new_task_leads = ['i'] * ntasks        
            if 'e2h' in order: #easy to hard path
                new_task_class_pairs = ['10-11','8-9','4-5','2-3','0-1','6-7']
            elif 'h2e' in order: #hard to easy path 
                new_task_class_pairs = ['6-7','0-1','2-3','4-5','8-9','10-11']
            new_task_fractions = [fraction] * ntasks
            
    return task_epochs, new_task_datasets, new_task_modalities, new_task_leads, new_task_class_pairs, new_task_fractions
    
def obtain_dataset_order(trial,cl_scenario,dataset_name,fraction,order):
    """ Obtain Relevant Information for Specified Continual Learning Scenario """
    new_task_batch_size = []
    new_task_held_out_lr = []

    if cl_scenario == 'Domain-IL':
        if dataset_name == 'fetal':
            task_epochs = 20
            ntasks = 4
            new_task_datasets = ['fetal'] * ntasks
            new_task_modalities = [['ecg']] * ntasks
            new_task_leads = ['Abdomen_1','Abdomen_2','Abdomen_3','Abdomen_4'] #couldnt't train anything but Abdomen_1
            new_task_class_pairs = [''] * ntasks
            new_task_fractions = [fraction] * ntasks
        elif dataset_name == 'ptb':
            task_epochs = 40 #40
            ntasks = 12
            new_task_datasets = ['ptb'] * ntasks
            new_task_modalities = [['ecg']] * ntasks
            new_task_leads = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
            new_task_class_pairs = [''] * ntasks
            new_task_fractions = [fraction] * ntasks
        elif dataset_name == 'physionet2020':
            task_epochs = 40#40 
            ntasks = 12
            new_task_datasets = [dataset_name] * ntasks
            new_task_modalities = [['ecg']] * ntasks
            new_task_leads = ['I','II','III','aVL','aVR','aVF','V1','V2','V3','V4','V5','V6']
            new_task_class_pairs = [''] * ntasks
            new_task_fractions = [fraction] * ntasks
    elif cl_scenario == 'Task-IL':
        task_epochs = 60 #80
        #new_task_epochs = np.array([0,20,100])
        ntasks = 2
        new_task_datasets = ['chapman','cardiology']#,'cardiology','physionet2017']#cardiology']
        new_task_modalities = [['ecg']] * ntasks
        new_task_leads = ["['all']",'i']#"['II', 'aVR']"] 
        new_task_class_pairs = ['Term 1','']#[''] * ntasks
        new_task_fractions = [1, 0.9]#[fraction] * ntasks
    elif cl_scenario == 'Class-IL':
        task_epochs = 20
        ntasks = 6
        new_task_datasets = ['cardiology'] * ntasks
        new_task_modalities = [['ecg']] * ntasks
        new_task_leads = ['i'] * ntasks        
        new_task_class_pairs = ['0-1','2-3','4-5','6-7','8-9','10-11']
        new_task_fractions = [fraction] * ntasks
    elif cl_scenario == 'Time-IL':
        task_epochs = 20
        ntasks = 3
        new_task_datasets = ['chapman'] * ntasks
        new_task_modalities = [['ecg']] * ntasks
        new_task_leads = ["['all']"] * ntasks
        new_task_class_pairs = ['Term 1','Term 2','Term 3']
        new_task_fractions = [1] * ntasks
        
    if trial == 'multi_task_learning':
        new_task_datasets,new_task_modalities,new_task_leads,new_task_class_pairs,new_task_fractions = modify_dataset_order_for_multi_task_learning(new_task_datasets,new_task_modalities,new_task_leads,new_task_class_pairs,new_task_fractions)
        new_task_epochs = [0]
        new_task_batch_size = [data2bs_dict[dataset_name]]
        new_task_held_out_lr = [data2lr_dict[dataset_name]]
        max_epochs = [200]
    else:
        """ Shuffle Order of Tasks """
        if isinstance(order,int):
            if order > 0:
                random.seed(order)
                indices = random.sample(list(np.arange(ntasks)),ntasks)
                new_task_datasets = [new_task_datasets[index] for index in indices]
                new_task_modalities = [new_task_modalities[index] for index in indices]
                new_task_leads = [new_task_leads[index] for index in indices]
                new_task_class_pairs = [new_task_class_pairs[index] for index in indices]
                new_task_fractions = [new_task_fractions[index] for index in indices]
        elif isinstance(order,str):
            if 'curriculum' in order:
                task_epochs, new_task_datasets, new_task_modalities, new_task_leads, new_task_class_pairs, new_task_fractions = obtain_dataset_order_for_curriculum(trial,cl_scenario,dataset_name,fraction,order)
        
        if 'new_task_epochs' not in locals(): #if variable does not exist, create it
            new_task_epochs = np.arange(0,len(new_task_datasets)*task_epochs,task_epochs) 
        #new_task_fractions = [fraction] * len(new_task_epochs)    
        for dataset in new_task_datasets:
            new_task_batch_size.append(data2bs_dict[dataset])
            new_task_held_out_lr.append(data2lr_dict[dataset])
        
        max_epochs = max(new_task_epochs) + task_epochs
    
    return new_task_datasets, new_task_modalities, new_task_leads, new_task_epochs, new_task_fractions, new_task_batch_size, new_task_held_out_lr, new_task_class_pairs, max_epochs
#%%
def obtain_dicts(new_task_datasets, new_task_modalities, new_task_leads, new_task_epochs, new_task_fractions, new_task_batch_size, new_task_held_out_lr, new_task_class_pairs, downstream_task):
    """ Dict for Leads to Use - Only Affect PTB Datasets """
    new_task_leads_dict = dict(zip(new_task_epochs,new_task_leads))
    """ Dict for Transition Datasets and Corresponding Modality """
    new_task_modalities_dict = dict(zip(new_task_epochs,new_task_modalities))
    """ Dict For Transition Epochs and Transition Datasets """
    new_task_dict = dict(zip(new_task_epochs,new_task_datasets))
    """ Dict for Transition Datasets and Transition Labelled Fraction """
    new_task_fraction_dict = dict(zip(new_task_epochs,new_task_fractions))
    """ Dict for Transition Datasets and Corresponding Batch Size """
    new_task_batch_dict = dict(zip(new_task_epochs,new_task_batch_size))
    """ Dict for Transition Datasets and Corresponding Learning Rate """
    new_task_lr_dict = dict(zip(new_task_epochs,new_task_held_out_lr))
    """ Dict for Transition Epochs and Corresponding Class Pairs """
    new_task_class_pairs_dict = dict(zip(new_task_epochs,new_task_class_pairs))
    """ When to Perform Forward Passes on Storage Buffer """
    new_task_epochs = list(new_task_leads_dict.keys())
    if downstream_task == 'continual_buffer':
        first_epoch = new_task_epochs[1]
        acquisition_epochs = np.arange(first_epoch,np.max(new_task_epochs)+first_epoch,1) #[3,6]
        """ When to Sample and Train with Augmented Dataset """
        sample_epochs = np.arange(first_epoch+1,np.max(new_task_epochs)+first_epoch,1) #[5,10]
    else:
        acquisition_epochs = []
        sample_epochs = []
    """ How Many Tasks Back to Sample From """
    look_back = 2 #not currently implemented
    
    all_task_dict_names = ['new_task_leads_dict','new_task_modalities_dict','new_task_dict','new_task_fraction_dict','new_task_batch_dict','new_task_lr_dict','new_task_class_pairs_dict']
    all_task_dicts = new_task_leads_dict, new_task_modalities_dict, new_task_dict, new_task_fraction_dict, new_task_batch_dict, new_task_lr_dict, new_task_class_pairs_dict
    all_task_dict = dict(zip(all_task_dict_names,all_task_dicts))
    
    return all_task_dict, acquisition_epochs, sample_epochs, look_back
