#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:56:44 2020

@author: Dani Kiyasseh
"""

#%%
""" Functions in this script:
    1) retrieve_entropy
    2) retrieve_variance_ratio
    3) retreive_acquisition_metric
    4) retrieve_time_metric
    5) update_acquisition-dict
    6) obtain_aq_threshold
    7) select_sample_indices
    8) acquisition_function
    9) perform_MC_sampling
    10) obtain_output_probs
    11) obtain_prediction
    12) obtain_entropy_threshold
    13) change_ground_truth_label
    14) retrieve_gaussian_intersection
    15) condition_for_oracle
"""    
#%%

import torch
import os
from scipy.special import expit, softmax
from scipy.stats import entropy, norm
import numpy as np
from statistics import mode
from numpy import linalg
from operator import itemgetter
from sklearn.mixture import GaussianMixture
from collections import Counter
import random
                                    
from perform_training import one_epoch

#%%

def retrieve_entropy(classification,dataset,array): #array is 1xC 
    if classification == '2-way':
        array = expit(array)
        if dataset == 'physionet2020': #binary mutlilabel situation 
            np.save(os.path.join('/home/scro3517/Desktop','array'),array)            
            entropy_estimate = np.mean([entropy([el,1-el]) for el in array])
            #print(entropy_estimate)
        else:
            entropy_estimate = entropy([array,1-array])
    elif classification is not None and classification != '2-way':
        #print('Pre Softmax Vector')
        #print(array)
        array = softmax(array)
        entropy_estimate = entropy(array) #entropy also accepts logit values (it will normalize it)
    return entropy_estimate

def retrieve_variance_ratio(classification,array):
    if classification == '2-way':
        array = expit(array) 
        class_predictions = np.where(array>0.5,0,1)
    elif classification is not None and classification != '2-way':
        array = softmax(array)
        class_predictions = np.argmax(array,axis=1)

    num_passes = len(class_predictions)
    most_frequent_class = mode(class_predictions,axis=None).mode.item()
    fm = np.sum(class_predictions == most_frequent_class)
    var_ratio = 1-(fm/num_passes)
    
    return var_ratio

def retrieve_acquisition_metric(classification,posterior_dict,metric,dataset,perturbed_posterior_dict=None):#,acquisition_metric_dict_prev):
    metric = metric.split('.')[0] #to remove potential time part
    acquisition_metric_dict = dict()
    if perturbed_posterior_dict is None:
        for task_name,index_dict in posterior_dict.items():
            acquisition_metric_dict[task_name] = dict()
            """ Array represents the MC logit outputs for a particular input sample """
            for index,array in index_dict.items():
                #print(index)
                """ Take Average Posterior Probability Across Inference Runs for Same Index - Array Dim TxC """
                array = np.array(array) #TxC
                posterior_dist = np.mean(array,0) #1xC
                posterior_dist = np.float64(posterior_dist)
                if metric == 'entropy':
                    entropy_estimate = retrieve_entropy(classification,dataset,posterior_dist)
                    acquisition_metric = entropy_estimate
                elif metric == 'bald':
                    entropy_of_mixture = retrieve_entropy(classification,dataset,posterior_dist)
                    #print(entropy_of_mixture)
                    mixture_of_entropy = [] 
                    for mc_array in array:
                        entropy_of_mc = retrieve_entropy(classification,dataset,mc_array) #1xC argument
                        mixture_of_entropy.append(entropy_of_mc)
                    mixture_of_entropy = np.mean(mixture_of_entropy)
                    bald = entropy_of_mixture - mixture_of_entropy
                    acquisition_metric = bald
                elif metric == 'variance_ratio':
                    var_ratio = retrieve_variance_ratio(classification,array)            
                    acquisition_metric = var_ratio
                
                acquisition_metric_dict[task_name][index] = acquisition_metric
    else:
        print(len(posterior_dict),len(perturbed_posterior_dict))
        for (task_name,clean_dict),pert_dict in zip(posterior_dict.items(),perturbed_posterior_dict.values()):
            acquisition_metric_dict[task_name] = dict()
            for (index,clean_array),pert_array in zip(clean_dict.items(),pert_dict.values()):
                #print(clean_array)
                #print(pert_array)
                clean_array = np.array(clean_array)
                pert_array = np.array(pert_array)
                
                if classification == '2-way':
                    clean_array = np.expand_dims(clean_array,1)
                    pert_array = np.expand_dims(pert_array,1)
                    
                    clean_array = expit(clean_array)
                    pert_array = expit(pert_array)
                    
                    softmax_clean_array = np.concatenate((clean_array,1-clean_array),1)
                    softmax_pert_array = np.concatenate((pert_array,1-pert_array),1)
                else:
                    """ Softmax Needed for Entropy Function """
                    softmax_clean_array = softmax(clean_array,axis=1)
                    softmax_pert_array = softmax(pert_array,axis=1)
                
                mean_soft_clean_array = np.mean(softmax_clean_array,axis=0)
                mean_soft_pert_array = np.mean(softmax_pert_array,axis=0)
                #print(metric)
                if metric == 'balc_JSD':
                    #print(softmax_clean_array.shape)
                    #print(softmax_pert_array.shape)
                    #print('JSD')
                    mix_of_kld = np.mean(entropy(softmax_clean_array.transpose(),softmax_pert_array.transpose()))
                    kld_of_mix = entropy(mean_soft_clean_array,mean_soft_pert_array)
                    JSD = mix_of_kld - kld_of_mix
                    acquisition_metric = JSD
                elif metric == 'balc_KLD':
                    #print('KLD')
                    cov_soft_clean_array = np.cov(softmax_clean_array.transpose())
                    cov_soft_pert_array = np.cov(softmax_pert_array.transpose())
                    
                    try: #try branch in the event we get a singulalr matrix
                        term1 = np.trace(np.dot(linalg.inv(cov_soft_pert_array+1e-8),cov_soft_clean_array))
                        term2a = np.dot(mean_soft_pert_array - mean_soft_clean_array,linalg.inv(cov_soft_pert_array+1e-8))
                        term2 = np.dot(term2a,np.transpose(mean_soft_pert_array - mean_soft_clean_array))
                        term3 = np.log((linalg.det(cov_soft_pert_array)/(linalg.det(cov_soft_clean_array)+1e-8)))
                        kld_of_mvn = 0.5*(term1 + term2 + term3)
                    except:
                        kld_of_mvn = 0
                    acquisition_metric = kld_of_mvn
                
                acquisition_metric_dict[task_name][index] = acquisition_metric
    
    return acquisition_metric_dict

def retrieve_time_metric(cum_acquisition_metric_dict):
    print('TIME!')
    acquisition_metric_dict = dict()
    for task_name,index_dict in cum_acquisition_metric_dict.items():
        acquisition_metric_dict[task_name] = dict()
        for index,array in index_dict.items():
            acquisition_metric_dict[task_name][index] = np.trapz(array) #you need at least 2 epochs worth of data
        
    return acquisition_metric_dict

def update_acquisition_dict(dataset,epoch,metric,classification,posterior_dict,acquisition_metric_dict,full_dict_for_saving,acquired_indices,perturbed_posterior_dict=None):
    
    """ Current Acquisition Metric Dict """
    acquisition_metric_dict_current = retrieve_acquisition_metric(classification,posterior_dict,metric,dataset,perturbed_posterior_dict)
    #print(len(acquisition_metric_dict_current))
    #print(list(acquisition_metric_dict_current.keys()))
    
    """ If First Epoch, Prepare New Dict """
    if epoch == 0:
        acquisition_metric_dict = dict()
        full_dict_for_saving = dict()
        for task_name,index_dict in acquisition_metric_dict_current.items():
            acquisition_metric_dict[task_name] = {index:[] for index in index_dict.keys()}
            full_dict_for_saving[task_name] = {index:[] for index in index_dict.keys()}
        #print(acquisition_metric_dict)
    
    """ Build Acquisition Metric Dict As A Function of Epochs """
    for task_name,index_dict in acquisition_metric_dict_current.items():
        for index in index_dict.keys():
            acquisition_metric_dict[task_name][index].append(index_dict[index])
            full_dict_for_saving[task_name][index].append(index_dict[index])
    
    """ Remove Acquired Indices Entries from Acquisition Dict To Avoid Choosing Already Acquired Index """
    remaining_acquisition_metric_dict = dict()
    for task_name,index_dict in acquisition_metric_dict.items():
        keep_indices = list(set(index_dict.keys()) - set(acquired_indices))    
        remaining_acquisition_metric_dict[task_name] = dict(zip(keep_indices,list(itemgetter(*keep_indices)(index_dict))))
        
    return remaining_acquisition_metric_dict, full_dict_for_saving

def obtain_aq_threshold(acquisition_metric_dict):
    """ EVT-Inspired Threshold for Aq Function Values """
    #print(acquisition_metric_dict.values())
    aq_values = np.fromiter(acquisition_metric_dict.values(),dtype=float)
    print(aq_values)
    min_value = np.min(aq_values)
    aq_values = aq_values - min_value # to make sure lower bound is 0
    umax = np.max(aq_values)
    #print(umax)
    us = np.linspace(0,umax,100)
    mean_excess = []
    for u in us:
        residuals = aq_values - u
        pos_residuals = residuals[residuals>0]
        mean_residual = np.mean(pos_residuals)
        mean_excess.append(mean_residual)
    
    print(mean_excess)
    gradient = np.diff(mean_excess)
    for i,el in enumerate(gradient[:-1]):
        if np.sign(gradient[i+1]) != np.sign(gradient[i]):
            index = i
            threshold = us[index] - min_value
    
    print(threshold)
    return threshold

def select_sample_indices(selection_metric,nsamples_unlabelleld,samples_to_acquire,acquisition_metric_dict):#,highest=True):
    """ Obtain Indices Based on Acquisition Metric """   
#    if highest == True: #acquire highest valued samples
#        reverse = True
#    else: #acquire lowest valued samples
#        reverse = False
        
    indices = dict()
    for task_name,metric_dict in acquisition_metric_dict.items():
        if selection_metric == 'percentage':
            #print(acquisition_metric_dict)
            indices[task_name] = list(dict(sorted(metric_dict.items(),key=lambda x:x[1],reverse=True)[:samples_to_acquire[task_name]]).keys())
#        elif selection_metric == 'EVT':
#            """ Extreme Value Theorem Based Threshold """
#            threshold = obtain_aq_threshold(acquisition_metric_dict)
#            indices = [ind for ind,val in acquisition_metric_dict.items() if val > threshold]
#            print(len(indices))
#        elif selection_metric == 'random':
#            indices = random.sample(list(acquisition_metric_dict.keys()),samples_to_acquire)
        
    return indices    

def acquisition_function(dataset,save_path_dir,epoch,seed,metric,posterior_dict,modality_dict,gt_labels_dict,acquired_indices,acquired_prediction_dict,acquired_modality_dict,acquired_gt_labels_dict,classification,acquisition_percent=0.02,acquisition_metric_dict=None,perturbed_posterior_dict=None,task_names_dict=None,trial=None,abstention_threshold=0,hellinger=0,oracle_asks=[],noise_type=None,noise_level=0,nn_labels=None,tolerance=None,proportion_wasted=[]):#,highest=True):

    print('Retrieving Acquisition Metric')
    if 'time' not in metric:
        acquisition_metric_dict = retrieve_acquisition_metric(classification,posterior_dict,metric,dataset,perturbed_posterior_dict)
    elif 'time' in metric:
        acquisition_metric_dict = retrieve_time_metric(acquisition_metric_dict)
    
    acquisition_percent_list = acquisition_percent  #list of how much to acquire from each task
    #print('Acquisition Percent: %.3f' % acquisition_percent)
    
    samples_to_acquire = dict()
    for (task_name,index_dict),acquisition_percent in zip(posterior_dict.items(),acquisition_percent_list): 
        nsamples_unlabelled = len(index_dict) #keys are task_name
        samples_to_acquire[task_name] = int(nsamples_unlabelled*acquisition_percent)
    #print('Samples to Acquire')
    #print(samples_to_acquire)
    #""" Select Sample Indices """
    selection_metric = 'percentage' # 'random' OR 'EVT' OR 'percentage'
    #print('Selection Metric!')
    #print(selection_metric)
    print('Selecting Indices')
    indices = select_sample_indices(selection_metric,nsamples_unlabelled,samples_to_acquire,acquisition_metric_dict)#,highest=highest) #indices is a dict

    """ Obtain Predictions from Acquired Indices """
    acquired_posterior_dict = dict() #in case it does NOT get populated
    for task_name,task_indices in indices.items():
        if len(task_indices) > 1:
            elements = itemgetter(*task_indices)(posterior_dict[task_name])
            acquired_posterior_dict[task_name] = dict(zip(task_indices,elements))
        elif len(task_indices) == 1:
            index = task_indices[0]
            elements = posterior_dict[task_name][index]
            acquired_posterior_dict[task_name] = {index:elements}
        #print(elements)
    
    """ Obtain Ground Truth Labels """
    mods = dict()
    gt_labels = dict()
    acquired_modality_dict = dict()
    for task_name,task_indices in indices.items():  
        acquired_modality_dict[task_name] = dict()
        if len(task_indices) > 0:
            if len(task_indices) > 1: #itemgetter works with multiple indices
                mods[task_name] = list(map(lambda x: x[0],itemgetter(*task_indices)(modality_dict[task_name])))
                gt_labels[task_name] = list(map(lambda x: x[0],itemgetter(*task_indices)(gt_labels_dict[task_name])))
            elif len(task_indices) == 1: #if only single index chosen
                mods[task_name] = modality_dict[task_name][indices[0]]
                gt_labels[task_name] = gt_labels_dict[task_name][indices[0]]
            
            acquired_modality_dict[task_name][epoch] = dict(Counter(mods[task_name]))
            if dataset == 'physionet2020':
                acquired_gt_labels_dict[task_name][epoch] = np.nonzero(gt_labels[task_name])[0]
            else:
                acquired_gt_labels_dict[task_name][epoch] = dict(Counter(gt_labels[task_name]))
    
    """ Obtain Acquired Labels from Network Predictions """
    prediction_dict = dict()
    oracle_ask_fraction = 0
    false_positive_area = []
    print('Acquiring Labels')
    for task_name,index_dict in acquired_posterior_dict.items():
        prediction_dict[task_name] = dict()
        for index,list_of_probs in index_dict.items():
            #print(list_of_probs)
            #""" Added New """
            #list_of_probs = obtain_output_probs(list_of_probs[:,:-1],classification)
            #""" Ended """
            #posterior_dist = np.mean(list_of_probs,0) #list of probs across MC samples for ONE instance 
            """ Ground Truth """
            if dataset == 'physionet2020':
                ground_truth_label = gt_labels_dict[task_name][index]
            else:
                ground_truth_label = gt_labels_dict[task_name][index][0]
            """ Noise Applied to Labels """
            ground_truth_label = change_ground_truth_label(index,ground_truth_label,nn_labels,classification,noise_type,noise_level,epoch,seed)
            
            if trial == 'abstention_penalty':
                """ Convert Logits to Probs """
                #print(np.array(list_of_probs))
                array_of_probs = np.array(list_of_probs)
                #print(array_of_probs.shape)
                class_probs = softmax(array_of_probs[:,:-1],1)
                abstention_probs = expit(array_of_probs[:,-1])
                """ Take Average Across MC Samples """
                posterior_dist = np.mean(class_probs,0) #list of probs across MC samples for ONE instance 
                abstention_prob = np.mean(abstention_probs,0)
    
                #abstention_prob = posterior_dist[-1]
                #posterior_dist = posterior_dist[:-1]
                """ Use Class Probs to Get Class Prediction """
                prediction = obtain_prediction(posterior_dist,classification)
                """ We can change this threshold """
                #print('Prob of Asking Oracle')
                #print(abstention_prob)
                #print(gt_labels_dict[index][0],prediction)
                
                condition,false_positive_area = condition_for_oracle(abstention_threshold,abstention_prob,tolerance,false_positive_area)
                
                if hellinger > 0.15:
                    if condition: 
                        prediction_dict[task_name][index] = ground_truth_label
                        oracle_ask_fraction += 1
                    else:
                        prediction_dict[task_name][index] = prediction 
                else: #conservative default 
                    prediction_dict[task_name][index] = ground_truth_label
                    oracle_ask_fraction += 1
            elif 'e-greedy' in trial:
                """ Decaying Dependence on Oracle """
                array_of_probs = np.array(list_of_probs)
                #print(array_of_probs.shape)
                class_probs = softmax(array_of_probs,1)
                """ Take Average Across MC Samples """
                posterior_dist = np.mean(class_probs,0) #list of probs across MC samples for ONE instance 
                """ Use Class Probs to Get Class Prediction """
                prediction = obtain_prediction(posterior_dist,classification)
                epsilon = np.exp(-epoch/25) #after 5 acquisitions, assuming each acquisition it at 
                np.random.seed((index+1)*(epoch+1)*(seed+1))
                rand = np.random.uniform(0,1)
                if rand <= epsilon: #ask oracle #exponentially decay dependence on oracle
                    prediction_dict[task_name][index] = ground_truth_label
                else: #otherwise use prediction 
                    prediction_dict[task_name][index] = prediction  
            elif 'softmax_response' in trial: #needs some modification
                #nclasses = int(classification.split('-')[0])
                """ Threshold Choice Will Depend on Number of Classes in Classification """
                threshold = 0.5 #the higher, the more conservative the strategy is (more reliance on oracle)
                array_of_probs = np.array(list_of_probs)
                #print(array_of_probs.shape)
                class_probs = softmax(array_of_probs,1)
                """ Take Average Across MC Samples """
                posterior_dist = np.mean(class_probs,0) #list of probs across MC samples for ONE instance 
                """ Use Class Probs to Get Class Prediction """
                prediction = obtain_prediction(posterior_dist,classification)
                if np.max(posterior_dist) > threshold: # if confidence based on softmax response exists, take net prediction
                    prediction_dict[task_name][index] = prediction
                else:
                    prediction_dict[task_name][index] = ground_truth_label
            elif 'entropy_response' in trial:
                threshold = obtain_entropy_threshold(classification) #lower threshold means more dependence 
                array_of_probs = np.array(list_of_probs)
                class_probs = softmax(array_of_probs,1)
                posterior_dist = np.mean(class_probs,0)
                entropy_value = entropy(posterior_dist)
                prediction = obtain_prediction(posterior_dist,classification)
                if entropy_value > threshold:
                    prediction_dict[task_name][index] = ground_truth_label
                else:
                    prediction_dict[task_name][index] = prediction
            elif 'gt' in trial:
                """ CAUTION! - Labels Based on Ground Truth - Use for Checking Abnormalities Only """
                prediction_dict[task_name][index] = ground_truth_label
            else:
                """ Labels Based on Network Predictions """
                array_of_probs = np.array(list_of_probs)
                list_of_probs = obtain_output_probs(array_of_probs,classification)
                posterior_dist = np.mean(list_of_probs,0)
                prediction = obtain_prediction(posterior_dist,classification)
                prediction_dict[task_name][index] = prediction            
    
#    if len(indices) > 0:
#        #print('Oracle Fraction!')
#        #print(oracle_ask_fraction/len(indices))
#        oracle_asks.append(oracle_ask_fraction/len(indices))
#        np.save(os.path.join(save_path_dir,'oracle_asks'),np.array(oracle_asks))
#        
#    if not isinstance(false_positive_area,list):
#        proportion_wasted.append(false_positive_area)
#        np.save(os.path.join(save_path_dir,'proportion_wasted'),np.array(proportion_wasted))

    """ Add Indices to Acquired Indices """
    for task_name,task_indices in indices.items():
        acquired_indices[task_name] += task_indices
    """ Add Modalities """
    #acquired_modality_dict = {**acquired_modality_dict,**acquired_modality_dict_new}
    """ Add Predictions to Prediction Dict """
    for task_name in indices.keys():
        acquired_prediction_dict[task_name] = {**acquired_prediction_dict[task_name],**prediction_dict[task_name]}
    
    if task_names_dict is not None:
        retrieval_buffer_dict = dict()
        for task_name,task_indices in indices.items():
            retrieval_buffer_dict[task_name] = task_indices
    else:
        retrieval_buffer_dict = None
    #acquired_modality_dict[epoch] = dict(Counter(mods))
    #acquired_gt_labels_dict[epoch] = dict(Counter(gt_labels))
    
    #torch.save(acquired_modality_dict,'acquired_modality_dict')
    torch.save(acquired_indices,os.path.join(save_path_dir,'acquired_indices_list'))
    #print('Acquired Indices')
    #print(acquired_indices)
    return acquired_indices,acquired_prediction_dict,acquired_modality_dict,acquired_gt_labels_dict,oracle_asks,proportion_wasted,retrieval_buffer_dict,samples_to_acquire

def perform_MC_sampling(dropout_samples,save_path_dir,seed,epoch_count,batch_size,fraction,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,aul_scaling_dict=None,inferences=None,acquired_indices=None,acquired_labels=None,input_perturbed=False,trial=None,leads='ii',lambda1=1):
#    if trial == 'ER-MIR':
#        dropout_samples = 1
        
    for i in range(dropout_samples):
        print('Variational Inference Round %i' % i)
        
        """ Control Dropout Mask """
        #if acquisition == 'deterministic':
        #    torch.manual_seed(0) #shouldnt affect initialization
        if acquisition == 'stochastic':
            """ Same Within Epoch (especially needed for BALC)
                Different Across MC Samples 
                Different Across Epochs 
                Different Across Seeds
            """
            torch.manual_seed((i+1)*(epoch_count+1)*(seed+1)) #seed ensures same weight in the case of perturbed inputs later but different across MC passes #for BALC
            #torch.manual_seed(0)
        
        """ Control Input Perturbation """
        if input_perturbed == True:
            if perturbation == 'deterministic':
                if i == 0: #i.e. only need to call this once
                    """ Both of These Seeds Are Valid - Experiment with Both """
                    ##np.random.seed(0) #replace 0 with 'seed' to allow for variety over seeds #constant seed means same perturbation across MC passes, changing seed means different perturbations 
                    """ Same Across MC Samples (especially needed for MC Consistency)
                        Different Across Epochs
                        Different Across Seeds
                    """
                    np.random.seed(i*(epoch_count+1)*(seed+1)) #same perturbation across MC samples but different across acquisition epochs
                    ##print_hyperparam_info(acquisition_epochs,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,dropout_samples,metric,batch_size,held_out_lr,seed)
                    """ Make Sure All Arguments Are Up-to-Date """
                    #dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,input_perturbed,leads=leads)
            elif perturbation == 'stochastic':
                np.random.seed(i*(epoch_count+1)*(seed+1))
                #np.random.seed(0)
                ##np.random.seed(i) #constant seed means same perturbation across MC passes, changing seed means different perturbations 
                #dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,input_perturbed,leads=leads)
        
        """ Perform Forward Pass """
        results_dictionary, outputs_list, labels_list, mix_coefs, modality_list, indices_list, task_names_list, scoring_function, hyperparam_dict = one_epoch(mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,aul_scaling_dict,trial=trial,epoch_count=epoch_count,lambda1=lambda1,save_path_dir=save_path_dir)
        #print(outputs_list[:5])
        gt_labels = np.concatenate(labels_list)
        modality_list = np.concatenate(modality_list)
        indices = np.concatenate(indices_list)
        task_names = np.concatenate(task_names_list)
        #print(task_names)
#        if trial == 'ER-MIR': #store losses for use later
#            posterior_dists = scoring_function #should be loss per instance Nx1
#        else:
        posterior_dists = np.concatenate(outputs_list)
        
        """ Dictionary for CL Buffer Scenario """
        unique_task_names = np.unique(task_names)
        posterior_dict = dict()
        task_names_dict = dict()
        modality_dict = dict()
        gt_labels_dict = dict()
        for task_name in unique_task_names:
            posterior_dict[task_name] = dict()
            task_names_dict[task_name] = dict()
            modality_dict[task_name] = dict()
            gt_labels_dict[task_name] = dict()
            
        """ Populate Distributions for Each Index """ #this works because indices are task-specific and not global
        for posterior_dist,task_name,index,modality,gt_label in zip(posterior_dists,task_names,indices,modality_list,gt_labels):
            posterior_dict[task_name][index] = posterior_dist
            task_names_dict[task_name][index] = task_name
            modality_dict[task_name][index] = modality
            gt_labels_dict[task_name][index] = gt_label
        
        #generate dicts for populating later
        if i == 0:
            posterior_dict_new = dict()
            modality_dict_new = dict()
            gt_labels_dict_new = dict()
            task_names_dict_new = dict()
            for task_name in unique_task_names:
                posterior_dict_new[task_name] = {index:[] for index in posterior_dict[task_name].keys()}
                modality_dict_new[task_name] = {index:[] for index in modality_dict[task_name].keys()}
                gt_labels_dict_new[task_name] = {index:[] for index in gt_labels_dict[task_name].keys()}
                task_names_dict_new[task_name] = {index:[] for index in task_names_dict[task_name].keys()}

        """ Accumulate Posterior Dists From Each Dropout Pass For Each Sample """
        for task_name in unique_task_names:
            for index in posterior_dict[task_name].keys():
                posterior_dict_new[task_name][index].append(posterior_dict[task_name][index])
                if i == 0:
                    modality_dict_new[task_name][index].append(modality_dict[task_name][index])
                    gt_labels_dict_new[task_name][index].append(gt_labels_dict[task_name][index])
                    task_names_dict_new[task_name][index].append(task_names_dict[task_name][index])
    #print('Nsamples in 1 MC Pass: %i' % len(indices))
    #print('Max Index in Indices: %i' % np.max(indices))
    return posterior_dict_new,modality_dict_new,gt_labels_dict_new,task_names_dict_new

def obtain_output_probs(posterior_dist,classification):
    if classification == '2-way':
        posterior_dist = expit(posterior_dist) 
        #prediction = np.where(posterior_dist>0.5,1,0).item()
    elif classification is not None and classification != '2-way':
        """ Added New Feb 3 """
        posterior_dist = softmax(posterior_dist,1) 
        """ Ended """
        #prediction = np.argmax(posterior_dist) #labels to assign to sample 
    return posterior_dist

def obtain_prediction(posterior_dist,classification):
    if classification == '2-way':
        prediction = np.where(posterior_dist>0.5,1,0).item()
    elif classification is not None and classification != '2-way':
        prediction = np.argmax(posterior_dist) #labels to assign to sample
    return prediction

def obtain_entropy_threshold(classification):
    if classification == '2-way':
        max_entropy = entropy([0.5,0.5])
    elif classification is not None and classification != '2-way':
        nclasses = int(classification.split('-')[0])
        max_entropy = entropy([1/nclasses for _ in range(nclasses)])
    entropy_threshold = 0.9*max_entropy
    return entropy_threshold

def change_ground_truth_label(index,ground_truth_label,nn_labels,classification,noise_type,noise_level,epoch,seed):
    """ Introduce Noise to Individal Label @ Different Probability Values """
    np.random.seed((index+1)*(epoch+1)*(seed+1))
    rand = np.random.uniform(0,1)
    if noise_type is not None and rand <= noise_level:
        original_label = ground_truth_label
        nclasses = int(classification.split('-')[0])
        class_set = set(np.arange(nclasses))
        remaining_class_set = list(class_set - set([original_label]))
        if noise_type == 'random':
            random.seed((epoch+1)*(seed+1))
            ground_truth_label = random.sample(remaining_class_set,1)[0]
        elif noise_type == 'nearest_neighbour':
            ground_truth_label = nn_labels[index]
                
    return ground_truth_label

def retrieve_gaussian_intersection(m1,m2,std1,std2):
    """ m2 > m1 """
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])

def condition_for_oracle(abstention_threshold,abstention_prob,tolerance,false_positive_area):
    gmm1 = abstention_threshold['gmm1']
    gmm2 = abstention_threshold['gmm2']
    if tolerance is None:
        """ OPTION 1 ----- Relative Value of Gaussians Thresholding """
        if isinstance(gmm1,GaussianMixture):
            abstention_prob = np.array(abstention_prob) #scalar value
            prob1 = gmm1.score_samples(abstention_prob.reshape(-1,1)) #scalar
            prob2 = gmm2.score_samples(abstention_prob.reshape(-1,1)) #scalar
            condition = prob2 > prob1
            
            mean1,var1 = gmm1.means_.item(),gmm1.covariances_.item()
            mean2,var2 = gmm2.means_.item(),gmm2.covariances_.item()
            intersect = retrieve_gaussian_intersection(mean1,mean2,np.sqrt(var1),np.sqrt(var2))
            lower_area = norm.cdf(intersect,loc=mean1,scale=np.sqrt(var1))
            false_positive_area = 1-lower_area
        else:
            condition = True #depend on oracle 
            false_positive_area = 1 #filler
        #print('Comparing Gaussians!')
    else:
        """ OPTION 2 ----- Tolerance-Dependent Thresholding on Output of Selection Function """
        mean,var = gmm1.means_.item(),gmm1.covariances_.item()
        quantile = 1-tolerance
        threshold = norm.ppf(quantile, loc=mean, scale=np.sqrt(var))
        condition = abstention_prob > threshold
        false_positive_area = 1 #filler
        #print('Enforcing Tolerance Constraint!')
    
    return condition,false_positive_area