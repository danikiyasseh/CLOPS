#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:51:29 2020

@author: Dani Kiyasseh
"""

import numpy as np
from prepare_network import cnn_network_time
from primarynetwork import PrimaryNetwork
from prepare_tasks import obtain_dicts, obtain_dataset_order
from run_experiment import make_saving_directory_continual, train_model

#%%
""" Directory to Folder with Datasets """
basepath_to_data = '/mnt/SecondaryHDD' #'/home/scro3517/Desktop' 
""" Define Downstream Task """
visualize_loss = False #filler for now 
alpha = 1 #filler for now 
""" Original Split According to Patients """
#fraction = 0.9
""" Of the Above, Subsample Labelled Data @ Frame-Level """
labelled_fraction = 1 #0.1
""" Subsample Unlabelled Data @ Frame-Level """
unlabelled_fraction = 1 #0.05
""" Initialization """
meta = False #use initialization from meta-training? #False allows you to compare to random initialization
#%%
def obtain_formulation_dict(formulation_of_interest,acquisition_func):
    #metric_list = ['bald']
    #balc_metric_list = ['balc_KLD']
    formulation_dict = {'mc_dropout':
                            {'acquisition': 'stochastic',
                             'input_perturbed': False,
                             'perturbation': 'deterministic',
                             'metric': acquisition_func},
                        'mc_consistency':
                            {'acquisition': 'deterministic',
                             'input_perturbed': True,
                             'perturbation': 'stochastic',
                             'metric': acquisition_func},
                        'balc':
                            {'acquisition': 'stochastic',
                             'input_perturbed': True,
                             'perturbation': 'deterministic',
                             'metric': acquisition_func}}
    
    return formulation_dict[formulation_of_interest]

#%%
""" IMPORTANT ---------- Change These to Control Phases """
phases = ['train','val']
if 'val' in phases and len(phases) == 1 or 'test' in phases and len(phases) == 1:
    saved_weights_list = ['finetuned_weight']
else:
    saved_weights_list = [None]

#%%
#save_path_dir = make_saving_directory(first_dataset,first_fraction,first_modality,meta,acquisition_epochs,metric,seed,acquisition=None,input_perturbed=False,perturbation=None)
#torch.autograd.set_detect_anomaly(True)
""" Choose Network of Interest """
def obtain_network(cl_strategy):
    if cl_strategy == 'fine-tuning':
        network = cnn_network_time
        bptt_steps = None
    else:
        network = PrimaryNetwork
        bptt_steps = 1000 #change this accordingly
    return network, bptt_steps
#%%
cl_scenario_list = ['Class-IL']#,'Task-IL','Domain-IL'] #Class-IL'
dataset_name_list = ['cardiology']#,'cardiology']#,'-','physionet2020']
fraction = 0.9 #applied to all datasets
cl_strategy = 'fine-tuning' #fine-tuning, nonlocal (default), dual
mask_gradients = False #mask gradients using fisher or similarity for all tasks except the first
regularization_term = False #reg term added to loss for all tasks except the first

#trial = 'gt' #'random_acquisition' #'random_storage' | 'random_acquisition' | default = 'gt' which means others are not done.

trials_list = ['gt'] #random_storage','random_acquisition','multi_task_learning'] # 'gt' | 'random_storage' | 'random_acquisition' | 'multi_task_learning' | 'fine-tuning' | 'gt_monte_carlo' | 'gt_acquisition_function'
""" Define Acquisition Function to Use """
formulation_of_interest = 'mc_dropout' # 'mc_dropout' | 'mc_consistency' | 'balc'
acquisition_func = 'bald' # 'bald'
formulation_dict = obtain_formulation_dict(formulation_of_interest,acquisition_func)
""" Number of MC Samples """
dropout_samples_list = [20] #default
""" What Percentage of Previous Task to Store in Storage Buffer """
storage_percent_options = [0.25] #[0.1,0.25,0.5,1] #[0.25]
storage_percent_list = [[percent for _ in range(12)] for percent in storage_percent_options]
""" What Percentage of Retrieval Buffer to Acquire """
acquisition_percent_options = [0.5] #[0.1,0.25,0.5,1] #[0.5]
acquisition_percent_list = [[percent for _ in range(12)] for percent in acquisition_percent_options]
""" Seeds Per Experiment """
max_seed = 5
seeds_list = np.arange(max_seed) #use seed 10 for quick check experiments 
""" Order of Tasks """
order_list = [2] #int or str

heads_list = ['single'] #'single' | 'multi'
for heads in heads_list:
    for cl_scenario,dataset_name in zip(cl_scenario_list,dataset_name_list):
        for order in order_list:
            for storage_percent in storage_percent_list:
                for acquisition_percent in acquisition_percent_list:
                    for trial in trials_list:
                        
                        if trial == 'random_storage':
                            task_instance_importance = False
                            downstream_task = 'continual_buffer'
                            highest = True # shouldn't do anything since random storage
                        elif trial == 'random_acquisition':
                            task_instance_importance = True
                            downstream_task = 'continual_buffer'
                            highest = True
                        elif trial == 'fine-tuning':
                            task_instance_importance = False
                            downstream_task = 'los'
                            highest = True #filler
                        elif trial == 'multi_task_learning':
                            task_instance_importance = False
                            downstream_task = 'multi_task_learning'
                            highest = True #filler
                        elif trial == 'gt_monte_carlo':
                            task_instance_importance = True
                            downstream_task = 'continual_buffer'
                            highest = True
                            dropout_samples_list = [5,10,50] #for ablation study
                        else: #default setting which is our method
                            task_instance_importance = True #False #learnable parameters for each instance of each task that are used to weight their instance loss
                            downstream_task = 'continual_buffer'#'continual_buffer' #'los' | 'continual_buffer'
                            highest = True #default = True = store instances INTO buffer with highest task_instance values, False = lowest
                        
                        acquisition, perturbation, input_perturbed, metric = formulation_dict['acquisition'], formulation_dict['perturbation'], formulation_dict['input_perturbed'], formulation_dict['metric']
                        for dropout_samples in dropout_samples_list:
                            for seed in seeds_list:
                                new_task_datasets, new_task_modalities, new_task_leads, new_task_epochs, new_task_fractions, new_task_batch_size, new_task_held_out_lr, new_task_class_pairs, max_epochs = obtain_dataset_order(trial,cl_scenario,dataset_name,fraction,order)
                                all_task_dict, acquisition_epochs, sample_epochs, look_back = obtain_dicts(new_task_datasets, new_task_modalities, new_task_leads, new_task_epochs, new_task_fractions, new_task_batch_size, new_task_held_out_lr, new_task_class_pairs, downstream_task)
                                save_path_dir = make_saving_directory_continual(trial,phases,downstream_task,cl_scenario,cl_strategy,heads,dataset_name,order,task_instance_importance,acquisition_epochs,storage_percent,acquisition_percent,seed,max_seed,highest,dropout_samples,acquisition_func)
                                print(save_path_dir)
                                
                                if save_path_dir == 'do not train' and 'train' in phases:
                                    continue
                                
                                if save_path_dir == 'do not test':
                                    continue
                                
                                network, bptt_steps = obtain_network(cl_strategy)
                                train_model(basepath_to_data,dropout_samples,cl_scenario,trial,save_path_dir,network,cl_strategy,all_task_dict,seed,meta,metric,acquisition_epochs,sample_epochs,unlabelled_fraction,labelled_fraction,visualize_loss,alpha,saved_weights_list,phases,downstream_task,acquisition_percent=acquisition_percent,storage_percent=storage_percent,highest=highest,bptt_steps=bptt_steps,heads=heads,mask_gradients=mask_gradients,reg_term=regularization_term,task_instance_importance=task_instance_importance,acquisition=acquisition,input_perturbed=input_perturbed,perturbation=perturbation,mixture=False,weighted_sampling=False,num_epochs=max_epochs)
