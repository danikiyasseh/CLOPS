#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:46:09 2020

@author: Dani Kiyasseh
"""
#%%

""" Functions in this Script 
    1) change_lr
    2) change_weight_decay
    3) obtain_loss_function
    4) obtain_predictions
    5) determine_classification_setting
    6) save_config_weights
    7) save_statistics
    8) track_instance_params
    9) save_continual_stats
    10) obtain_martha_acc
    11) obtain_martha_bwt
    12) obtain_tstep_bwt
    13) obtain_lambda_bwt
"""

#%%
import os
import torch
import numpy as np
import torch.nn as nn
from operator import itemgetter
import copy

#%%

def change_lr(epoch_count,optimizer):
    """ Manually change (multiplicative) learning rate at pre-defined epochs """
    transition_epochs = None
    scale = 0.5
    if transition_epochs is not None:
        if epoch_count == transition_epochs[0]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*scale
                print('LR: %.5f' % param_group['lr'])

def change_weight_decay(epoch_count,optimizer):
    """ Manually change (additive) weight decay at pre-defined epochs """
    transition_epochs = None #[8]
    scale = 1e-1
    if transition_epochs is not None:
        if epoch_count == transition_epochs[0]:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = param_group['weight_decay'] + scale
                print('Weight Decay: %.5f' % param_group['weight_decay'])

def obtain_loss_function(phase,classification,dataloaders_list,pos_weight=1,imbalance_penalty=None):
    if classification is not None:
        nclasses = classification.split('-')[0]
    
    if 'train' in phase:
        """ Dataloader - Image-Based """ 
        #train_indices = dataloaders_list[0]['train'].batch_sampler.sampler.data_source.indices
        #all_outputs = dataloaders_list[0]['train'].batch_sampler.sampler.data_source.outputs

        all_outputs = dataloaders_list[0]['train1'].batch_sampler.sampler.data_source.label_array

        if imbalance_penalty == True:
            """ Obtain Weights for Optimizer (Class Imbalance) """            

            train_outputs = list(itemgetter(*train_indices)(all_outputs))
            val,bins = np.histogram(train_outputs,nclasses)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
            loss_weight = torch.tensor(max(val)/val,dtype=torch.float,device=device)
            """ Define Optimizer """
            if classification is not None and classification != '2-way':
                criterion = nn.CrossEntropyLoss(pos_weight=loss_weight)
                criterion_single = nn.CrossEntropyLoss(pos_weight=loss_weight,reduction='none')
            elif classification == '2-way':
                criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
                criterion_single = nn.BCEWithLogitsLoss(pos_weight=loss_weight,reduction='none')                
        else:
            if classification is not None and classification != '2-way':
                criterion = nn.CrossEntropyLoss()
                criterion_single = nn.CrossEntropyLoss(reduction='none')          
            elif classification == '2-way':
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
                criterion_single = nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(pos_weight)) 
            elif classification is None:
                criterion = nn.MSELoss()
                criterion_single = nn.MSELoss(reduction='none')
                
        """ Running Loss per Sample """
        keys = np.arange(len(all_outputs))
        values = [[] for _ in range(len(keys))]
        per_sample_loss_dict = dict(zip(keys,values))
        
        return per_sample_loss_dict, criterion, criterion_single
    else:
        if classification is not None and classification != '2-way':
            criterion = nn.CrossEntropyLoss()
            criterion_single = nn.CrossEntropyLoss(reduction='none')          
        elif classification == '2-way':
            criterion = nn.BCEWithLogitsLoss()
            criterion_single = nn.BCEWithLogitsLoss(reduction='none') 
        elif classification is None:
            criterion = nn.MSELoss()
            criterion_single = nn.MSELoss(reduction='none')
            
        return criterion, criterion_single

def obtain_predictions(output_probs,device,classification):
    if classification is not None and classification != '2-way':
        _,preds = torch.max(output_probs,1)
    elif classification == '2-way':
        """ May have to Subtract Mean from Outputs Before Taking Sigmoid """
        #preds = torch.where(torch.sigmoid(outputs)>0.5,torch.tensor(1,device=device),torch.tensor(0,device=device))
        preds = torch.where(output_probs>0.5,torch.tensor(1,device=device),torch.tensor(0,device=device))
    return preds

def determine_classification_setting(dataset_name,cl_scenario,trial):
    if dataset_name == 'physionet':
        classification = '5-way'
    elif dataset_name == 'bidmc':
        classification = '2-way'
    elif dataset_name == 'mimic': #change this accordingly
        classification = '2-way'
    elif dataset_name == 'cipa':
        classification = '7-way'
    elif dataset_name == 'cardiology':
        classification = '12-way'
        if trial != 'multi_task_learning':
            if cl_scenario == 'Class-IL':
                classification = '2-way'
    elif dataset_name == 'physionet2017':
        classification = '4-way'
    elif dataset_name == 'tetanus':
        classification = '2-way'
    elif dataset_name == 'ptb':
        classification = '2-way'
    elif dataset_name == 'fetal':
        classification = '2-way'
    elif dataset_name == 'physionet2016':
        classification = '2-way'
    elif dataset_name == 'physionet2020':
        classification = '2-way' #because binary multilabel
    elif dataset_name == 'chapman':
        classification = '4-way'
    elif dataset_name == 'cifar10':
        classification = '10-way'
    elif dataset_name == 'ptbxl':
        classification = '2-way' #because binary multilabel
    #print('Original Classification %s' % classification)
    return classification

def save_config_weights(save_path_dir,best_model_weights):
    torch.save(best_model_weights,os.path.join(save_path_dir,'finetuned_weight'))

def save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict):
    torch.save(acc_dict,os.path.join(save_path_dir,'%s_acc' % prefix))
    torch.save(loss_dict,os.path.join(save_path_dir,'%s_loss' % prefix))
    torch.save(auc_dict,os.path.join(save_path_dir,'%s_auc' % prefix))
    
def track_instance_params(epoch_count,task_instance_params_dict,tracked_instance_params_dict,current_task_info,new_task_epochs):
    """ Track Task-Instance Params During Training """
    task = current_task_info['current_task_dataset']
    modality = current_task_info['current_modality']
    leads = current_task_info['current_leads']
    fraction = current_task_info['current_fraction']
    class_pair = current_task_info['current_class_pair']
    current_name = '-'.join((task,modality[0],str(fraction),leads,class_pair))
    task_instance_params_dict_copy = copy.deepcopy(task_instance_params_dict)
    for name,param_list in task_instance_params_dict_copy.items():
        if name == current_name:
            if epoch_count in new_task_epochs:
                #tracked_instance_params_dict[name] = dict()
                tracked_instance_params_dict[name] = {index:[] for index in range(len(param_list))}
                
            for index,param in enumerate(param_list):
                #if epoch_count == 0:
                #    tracked_instance_params_dict[name][index] = []
                param = param.cpu().detach().item()
                tracked_instance_params_dict[name][index].append(param)
                #print(tracked_instance_params_dict[name][index])
    return tracked_instance_params_dict,current_name

def save_continual_stats(save_path_dir,ave_dicts):
    for dict_name,dict_entry in ave_dicts.items():
        torch.save(dict_entry,os.path.join(save_path_dir,dict_name))
    print('Continual Dicts Saved!')

def obtain_martha_acc(metric):
    """ Obtain Acc as Described in Martha ICLR 2020 Paper """
    validation_keys = [key for key in metric.keys() if 'val' in key]
    final_values = []
    for key in validation_keys:
        print(key)
        print(metric[key])
        final_value = metric[key][-1]
        #final_value = final_value.cpu().detach().numpy()
        final_values.append(final_value)
    #ave_value = np.mean(final_values) 
    return final_values

def obtain_martha_bwt(metric,new_task_epochs):
    """ Obtain BWT as Described in Martha ICLR 2020 Paper """    
    validation_keys = [key for key in metric.keys() if 'val' in key][:-1]
    task_epochs = new_task_epochs[1:]
    diff = np.diff(new_task_epochs)[0]
    bwt_values = []
    for epoch,key in zip(task_epochs,validation_keys):
        Rin = metric[key][-1]
        Rii = metric[key][diff-1]
        bwt = Rin - Rii
        #bwt = bwt.cpu().detach().numpy()
        bwt_values.append(bwt)
    #ave_bwt = np.mean(bwt_values)
    return bwt_values

def obtain_tstep_bwt(metric,new_task_epochs,step=1):
    """ Average t-Step BWT for All Tasks """
    validation_keys = [key for key in metric.keys() if 'val' in key][:-1] #all but last b/c you cant quantify forgetting for last task as no tasks follow it
    task_epochs = new_task_epochs[1:]
    diff = np.diff(new_task_epochs)[0]
    bwt_values = []
    for epoch,key in zip(task_epochs,validation_keys):
        Rit = metric[key][diff-1 + diff*(step)]
        Rii = metric[key][diff-1]
        bwt = Rit - Rii
        #bwt = bwt.cpu().detach().numpy()
        bwt_values.append(bwt)
    #ave_bwt = np.mean(bwt_values)
    return bwt_values

def obtain_lambda_bwt(metric,new_task_epochs):
    """ Average t-Step BWT for All Steps and All Tasks """
    validation_keys = [key for key in metric.keys() if 'val' in key][:-1]
    task_epochs = new_task_epochs[1:]
    diff = np.diff(new_task_epochs)[0]
    bwt_values = []
    for epoch,key in zip(task_epochs,validation_keys):
        current_epoch_index = np.where([epoch == ep for ep in task_epochs])[0][0]
        steps = len(task_epochs) - current_epoch_index
        for step in range(1,steps):
            Rit = metric[key][diff-1 + diff*(step)]
            Rii = metric[key][diff-1]
            bwt = Rit - Rii
            #bwt = bwt.cpu().detach().numpy()
            bwt_values.append(bwt)
    #ave_bwt = np.mean(bwt_values)   
    return bwt_values
