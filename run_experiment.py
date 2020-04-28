#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:46:02 2020

@author: Dani Kiyasseh

"""
#%%
""" Functions in this script:
    1) make_saving_directory_continual
    2) make_dir
    3) train_model
    4) perform_MC_or_normal_training
"""
#%%

import os
import numpy as np
import copy
import torch
from prepare_acquisition_functions import perform_MC_sampling, acquisition_function
from prepare_miscellaneous import determine_classification_setting, track_instance_params, save_continual_stats, obtain_martha_acc, obtain_martha_bwt, obtain_tstep_bwt, obtain_lambda_bwt
from prepare_models import load_initial_model
from prepare_dataloaders import obtain_dataloaders_information
from prepare_miscellaneous import change_lr, change_weight_decay, obtain_loss_function, save_config_weights, save_statistics
from prepare_buffer import obtain_random_retrieval_buffer_dict, obtain_indices_for_buffer, obtain_random_storage_indices
from perform_training import one_epoch

#%%
def perform_MC_or_normal_training(dropout_samples,seed,batch_size,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,inferences,acquired_indices,acquired_labels,input_perturbed,perturbed_dataloaders_list=None,bptt_steps=None,epoch_count=None,new_task_epochs=None,save_path_dir=None,mask_gradients=False,current_task_info=None,task_instance_params_dict=None,fraction=1):
    if 'train2' in phase and inference == 'query':#True:
        """ Perform Inference T Times i.e. MC Dropout Implementation """
        
        if acquisition == 'stochastic':
            print('Clean Input MC')
            #posterior_dict_new, modality_dict_new, gt_labels_dict_new, task_names_dict_new = perform_MC_sampling(phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device)
            posterior_dict_new, modality_dict_new, gt_labels_dict_new, task_names_dict_new = perform_MC_sampling(dropout_samples,save_path_dir,seed,epoch_count,batch_size,fraction,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device)
            if input_perturbed == True:
                """ Perturbed Input Path """
                print('Perturbed Input MC')
                perturbed_posterior_dict_new, perturbed_modality_dict_new, _, task_names_dict_new = perform_MC_sampling(dropout_samples,save_path_dir,seed,epoch_count,batch_size,fraction,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,perturbed_dataloaders_list,models_list,mix_coefs,optimizer,device,input_perturbed=input_perturbed)#,trial=trial,leads=leads)
            else:
                perturbed_posterior_dict_new = None 

        elif acquisition == 'deterministic':
            print('MC Consistency!')
            posterior_dict_new, modality_dict_new, gt_labels_dict_new, task_names_dict_new = perform_MC_sampling(dropout_samples,save_path_dir,seed,epoch_count,batch_size,fraction,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,perturbed_dataloaders_list,models_list,mix_coefs,optimizer,device,input_perturbed=input_perturbed)#acquired_indices,acquired_labels,input_perturbed)
            perturbed_posterior_dict_new = None
        
        return posterior_dict_new, perturbed_posterior_dict_new, modality_dict_new, gt_labels_dict_new, task_names_dict_new
    else:
        """ Function to Run Training """
        results_dictionary, outputs_list, labels_list, mix_coefs, modality_list, indices_list, task_names_list, loss_list, hyperparam_dict = one_epoch(mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,bptt_steps=bptt_steps,epoch_count=epoch_count,new_task_epochs=new_task_epochs,save_path_dir=save_path_dir,mask_gradients=mask_gradients,current_task_info=current_task_info,task_instance_params_dict=task_instance_params_dict)
        return results_dictionary, outputs_list, labels_list, mix_coefs, modality_list, indices_list, task_names_list, loss_list

def train_model(basepath_to_data,dropout_samples,cl_scenario,trial,save_path_dir,network,cl_strategy,all_task_dict,seed,meta,metric,acquisition_epochs,sample_epochs,unlabelled_fraction,labelled_fraction,visualize_loss,alpha,saved_weights_list,phases,downstream_task,acquisition_percent=0.02,storage_percent=0.10,highest=True,bptt_steps=None,heads='multi',mask_gradients=False,reg_term=False,task_instance_importance=False,acquisition=None,input_perturbed=False,perturbation=None,mixture=False,weighted_sampling=False,num_epochs=150):
    """ Training and Validation For All Epochs """
    
    new_task_leads_dict = all_task_dict['new_task_leads_dict']
    new_task_modalities_dict =  all_task_dict['new_task_modalities_dict']
    new_task_dict = all_task_dict['new_task_dict']
    new_task_fraction_dict = all_task_dict['new_task_fraction_dict']
    new_task_batch_dict = all_task_dict['new_task_batch_dict']
    new_task_lr_dict = all_task_dict['new_task_lr_dict']
    new_task_class_pairs_dict = all_task_dict['new_task_class_pairs_dict']
    
    new_task_epochs = list(new_task_dict.keys())
    new_task_datasets = list(new_task_dict.values())
    new_task_modalities = list(new_task_modalities_dict.values())
    new_task_leads = list(new_task_leads_dict.values())
    new_task_fraction = list(new_task_fraction_dict.values())
    new_task_class_pairs = list(new_task_class_pairs_dict.values())
    new_task_held_out_lr = list(new_task_lr_dict.values())
    
    """ Info Given to Instantiate Task-Instance Params """
    new_task_info = {'new_task_datasets':new_task_datasets,
                     'new_task_modalities':new_task_modalities,
                     'new_task_leads':new_task_leads,
                     'new_task_fractions':new_task_fraction,
                     'new_task_class_pairs':new_task_class_pairs}
    
    first_lr = new_task_held_out_lr[0]
    
    best_loss = float('inf')
    auc_dict = dict()
    acc_dict = dict()
    loss_dict = dict()
    if 'test' not in phases:
        phases = ['train1','val']
        inferences = [False,False]
    else:
        inferences = [False]
    
    if 'train1' in phases: #prepare train1 
        #print('TRAIN1')
        #print(phases[0])
        acc_dict['train1'] = []
        loss_dict['train1'] = []
        auc_dict['train1'] = []            
            
    if trial == 'multi_task_learning':
        first_dataset = new_task_datasets[0][0]
        for phase in phases:
            acc_dict[phase] = []
            loss_dict[phase] = []
            auc_dict[phase] = []
        
        patience = 10
        num_epochs = num_epochs[0]
    else:
        first_dataset = new_task_datasets[0]
        for dataset,modalities,leads,fraction,class_pair in zip(new_task_dict.values(),new_task_modalities_dict.values(),new_task_leads_dict.values(),new_task_fraction_dict.values(),new_task_class_pairs_dict.values()):
            phase = '_'.join(('val',dataset,modalities[0],str(fraction),leads,class_pair))
            print(phase)
            acc_dict[phase] = []
            loss_dict[phase] = []
            auc_dict[phase] = []

        task_epochs = np.diff(list(new_task_dict.keys()))[0]
        patience = task_epochs+10 #for early stopping criterion
        
    stop_counter = 0
    epoch_count = 0
    
    """ Needed for Rehearsal-Based Continual Learning Strategy """
    tracked_loss = dict()
    storage_buffer_dict = dict()
    retrieval_buffer_dict = dict()
    #if downstream_task == 'continual_buffer':
    #    """ When to Sample Data from the Buffer """
    #    sample_epochs = np.arange(list(new_task_dict.keys())[1]+5,list(new_task_dict.keys())[-1],5)
    #else:
    #    sample_epochs = []
    
    classification = determine_classification_setting(first_dataset,cl_scenario,trial)
    #classification not actually used in this first instance - so no worries 
    models_list,mix_coefs,optimizer,device,task_instance_params_dict = load_initial_model(basepath_to_data,meta,classification,visualize_loss,alpha,network,cl_strategy,phases,save_path_dir,saved_weights_list,first_lr,continual_setting=True,dataset_name=first_dataset,bptt_steps=bptt_steps,heads=heads,setting=cl_scenario,new_task_info=new_task_info,task_instance_importance=task_instance_importance,cl_scenario=cl_scenario,trial=trial)    
    #print(list(models_list[0].parameters())[0][0][0])
    """ Running List of Indices to Acquire During Training """
    acquired_indices = dict() #indices of the unlabelled data
    acquired_labels = dict() #network labels of the unlabelled data
    acquired_modalities = dict() #modalities of unlabelled data
    acquired_gt_labels = dict() #ground truth labels of the unlabelled for analysis later
    if 'time' in metric:
        acquisition_metric_dict = dict()
    
    """ Track Loss Weighting for Use in CL Later """
    tracked_instance_params_dict = dict()
    """ Removed b/c Redundant """
    #dataloaders_list,operations = load_initial_data(phases,classification,fraction,inferences,unlabelled_fraction,labelled_fraction,test_representation,test_order,test_colourmap,test_dim,test_task,batch_size,modality,acquired_indices,acquired_labels,downstream_task,modalities,downstream_dataset)
    dataloaders_list = [] #filler
    relevant_datasets = []
    #train_scoring_function = 0 #needed for WeightedSampler 
    while stop_counter <= patience and epoch_count < num_epochs:
    #for epoch in range(num_epochs):
        if 'train1' in phases or 'val' in phases:
            print('Epoch %i/%i' % (epoch_count,num_epochs-1))
            print('-' * 10)
            
            """ ESSENTIAL FOR CONTINUAL LEARNING - Transition to New Dataset Based on Epoch Number """
            if epoch_count in list(new_task_dict.keys()):
                stop_counter = 0 #reset stop counter 
                best_loss = float('inf') #reset best loss
                
                downstream_dataset = new_task_dict[epoch_count]
                fraction = new_task_fraction_dict[epoch_count]
                batch_size = new_task_batch_dict[epoch_count]
                #lr = new_task_lr_dict[epoch_count]
                modalities = new_task_modalities_dict[epoch_count]
                leads = new_task_leads_dict[epoch_count]
                class_pair = new_task_class_pairs_dict[epoch_count]
                current_task_info = {'current_task_dataset':downstream_dataset,
                                     'current_modality':modalities,
                                     'current_leads':leads,
                                     'current_fraction':fraction,
                                     'current_class_pair':class_pair}
                if downstream_task == 'continual_buffer':
                    current_name = '-'.join((downstream_dataset,modalities[0],str(fraction),leads,class_pair))                
                    acquired_gt_labels[current_name] = dict()
                    acquired_labels[current_name] = dict()
                    acquired_indices[current_name] = []
                """ Obtain Datasets Preceding Current One """
                #current_dataset_id = [index for index,count in enumerate(list(new_task_dict.keys())) if count == epoch_count]
                #previous_dataset_names = new_task_datasets[:current_dataset_id] 
                
                """ Load Model with Potential Network Changes Mid-Training """
#                models_list,optimizer = load_models_list_continual(epoch_count,new_task_epochs,cnn_network_time,device,models_list,optimizer,held_out_lr,downstream_dataset)
                """ Load DataLoader with Potential Augmentation Mid-Training """
#                print(models_list)
                #print(list(models_list[0].parameters())[0][0][0])

            if input_perturbed == True:
                relevant_datasets,phases,inferences,dataloaders_list,perturbed_dataloaders_list = obtain_dataloaders_information(basepath_to_data,acquisition_epochs,sample_epochs,new_task_epochs,metric,epoch_count,input_perturbed,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,class_pair,downstream_task,downstream_dataset,dataloaders_list,relevant_datasets,leads,storage_buffer_dict,retrieval_buffer_dict,heads,cl_scenario,new_task_info,trial=trial)
            elif input_perturbed == False:
                relevant_datasets,phases,inferences,dataloaders_list = obtain_dataloaders_information(basepath_to_data,acquisition_epochs,sample_epochs,new_task_epochs,metric,epoch_count,input_perturbed,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,class_pair,downstream_task,downstream_dataset,dataloaders_list,relevant_datasets,leads,storage_buffer_dict,retrieval_buffer_dict,heads,cl_scenario,new_task_info,trial=trial)
                perturbed_dataloaders_list = []
            
            """ Change LR mid-training """
            change_lr(epoch_count,optimizer)
            """ Change Weight Decay mid-training """
            change_weight_decay(epoch_count,optimizer)
        elif 'test' in phases:
            print('Test Set')
        
        """ ACTUAL TRAINING AND EVALUATION """
        #print(phases)
        for relevant_dataset,phase,inference in zip(relevant_datasets,phases,inferences):            
            if heads == 'single':
                if trial == 'multi_task_learning':
                    relevant_dataset = relevant_dataset[0] #b/c I provide list in MTL setting
                
                if relevant_dataset == 'physionet2020' or cl_scenario == 'Class-IL': #multilabel binary classification setting
                    classification = determine_classification_setting(relevant_dataset,cl_scenario,trial)
                else:
                    """ March 17th 2020 """
                    if trial == 'multi_task_learning':
                        all_new_task_datasets = list(map(lambda x:x[0],new_task_datasets))
                    else:
                        all_new_task_datasets = new_task_datasets
                    """ End """
                    classification_per_dataset = [determine_classification_setting(dataset,cl_scenario,trial) for dataset in all_new_task_datasets]
                    classification_per_dataset = [1 if classification == '2-way' else int(classification.split('-')[0]) for classification in classification_per_dataset]
                    classification = '-'.join((str(sum(classification_per_dataset)),'way'))
                    
                    #if cl_scenario == 'Class-IL':
                        
            else:
                classification = determine_classification_setting(relevant_dataset,cl_scenario,trial) #classification specific to dataset
            print(classification)
            #if statement to avoid reloading if same dataset
            #models_list,optimizer = load_models_list_continual(epoch_count,new_task_epochs,network,cl_strategy,device,models_list,optimizer,lr,relevant_dataset,bptt_steps=bptt_steps,heads=heads)
            
            if 'train' in phase:
                if relevant_dataset == 'mimic':# and 'train' in phase:
                    count,bins = np.histogram(dataloaders_list[0]['train1'].batch_sampler.sampler.data_source.label_array,2)
                    pos_weight = count[0]/count[1]
                    pos_weight = 1
                else:
                    pos_weight = 1
                #elif relevant_dataset == 'cipa':
                #    pos_weight = 4.82
                #    print(pos_weight)
            else:
                pos_weight = 1
            
            if 'train' in phase:
                [model.train() for model in models_list]
                per_sample_loss_dict, criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list,pos_weight) #criterion specific to dataset
                #per_sample_loss_dict, criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list)
            elif 'val' in phase or 'test' in phase:
                [model.eval() for model in models_list]
                criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list) #criterion specific to dataset
            #print(list(models_list[0].parameters())[0][0][0])
            
            #print([model.training for model in models_list])
            #criterion,criterion_single = obtain_criterion(phase,models_list,classification,dataloaders_list)
            
            if 'train2' in phase and inference == 'query':#True: #inference = query means peform MC sampling
                if trial != 'random_acquisition': #no need for MC is simply acquiring randomly and not based on acquisition function
                    if np.sum(acquisition_percent) < len(acquisition_percent): #if acquisition percent values are less than 1, then MC samples are worth doing
                        posterior_dict_new, perturbed_posterior_dict_new, modality_dict_new, gt_labels_dict_new, task_names_dict_new = perform_MC_or_normal_training(dropout_samples,seed,batch_size,modalities,relevant_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,inferences,acquired_indices,acquired_labels,input_perturbed,perturbed_dataloaders_list,epoch_count=epoch_count,save_path_dir=save_path_dir)
            else:
                #print(task_instance_params_dict)
                results_dictionary, outputs_list, labels_list, mix_coefs, modality_list, indices_list, task_names_list, loss_list = perform_MC_or_normal_training(dropout_samples,seed,batch_size,modalities,relevant_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,inferences,acquired_indices,acquired_labels,input_perturbed,bptt_steps=bptt_steps,epoch_count=epoch_count,new_task_epochs=new_task_epochs,save_path_dir=save_path_dir,mask_gradients=mask_gradients,current_task_info=current_task_info,task_instance_params_dict=task_instance_params_dict)
                """ Record Results """
                epoch_loss, epoch_acc, epoch_auroc = results_dictionary['epoch_loss'], results_dictionary['epoch_acc'], results_dictionary['epoch_auroc']
                
                """ Track Loss in train1 phase """
                if downstream_task == 'continual_buffer':
                    indices_list = np.concatenate(indices_list)
                    outputs_list = np.concatenate(loss_list)
                    if 'train' in phase:
                        """ Initialize Tracked Loss - Reset for Each New Task """
                        if epoch_count in list(new_task_dict.keys()):
                            tracked_loss = {index:[] for index in indices_list}
                        
                        current_loss = dict(zip(indices_list,outputs_list))
                        for index,loss in current_loss.items():
                            if index in tracked_loss.keys(): #only track indices of current task's data (not augmented)
                                tracked_loss[index].append(loss)
            
            """ Print Indices to Check Order """
            #if phase == 'train1':
            #    print(outputs_list)
            #    print(np.concatenate(outputs_list).shape)
            
            """ Acquisition of New Datapoints Based on Acquisition Function """
            if 'train2' in phase and inference == 'query' and 'time' not in metric: #remember train2 in this scenario will only happen if epoch is in acquisition_epochs
                #torch.save(posterior_dict_new,'posterior_dict')
                #torch.save(perturbed_posterior_dict_new,'perturbed_posterior_dict')
                #acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels, _ , _ ,retrieval_buffer_dict = acquisition_function(epoch_count,metric,posterior_dict_new,modality_dict_new,gt_labels_dict_new,acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,classification,perturbed_posterior_dict=perturbed_posterior_dict_new,task_names_dict=task_names_dict_new)
                if trial == 'random_acquisition':
                    if np.sum(acquisition_percent) < len(acquisition_percent):
                        retrieval_buffer_dict = obtain_random_retrieval_buffer_dict(storage_buffer_dict,acquisition_percent) #dict with keys = task_name, values = task_indices
                    else:
                        retrieval_buffer_dict = storage_buffer_dict
                else:
                    if np.sum(acquisition_percent) < len(acquisition_percent):
                        acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels, _ , _ ,retrieval_buffer_dict,samples_to_acquire = acquisition_function(relevant_dataset,save_path_dir,epoch_count,seed,metric,posterior_dict_new,modality_dict_new,gt_labels_dict_new,acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,classification,acquisition_percent=acquisition_percent,perturbed_posterior_dict=perturbed_posterior_dict_new,task_names_dict=task_names_dict_new,trial=trial) #,highest=highest)
                    else:
                        retrieval_buffer_dict = storage_buffer_dict
                        samples_to_acquire = {task_name:len(elements) for task_name,elements in storage_buffer_dict.items()} #b/c acquisition is 1, all storage elements are acquired

            elif 'train2' in phase and inference == 'query' and 'time' in metric:
                if np.sum(acquisition_percent) < len(acquisition_percent):
                    acquisition_metric_dict = update_acquisition_dict(relevant_dataset,epoch_count,classification,posterior_dict_new,acquisition_metric_dict,acquired_indices,perturbed_posterior_dict_new)
                    if epoch_count in acquisition_epochs:
                        #if len(acquired_indices) != total_unlabelled_samples:
                        #acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels, _ , _ ,retrieval_buffer_dict = acquisition_function(epoch_count,metric,posterior_dict_new,modality_dict_new,gt_labels_dict_new,acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,classification,acquisition_metric_dict=acquisition_metric_dict,task_names_dict=task_names_dict_new)
                        acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels, _ , _ ,retrieval_buffer_dict,samples_to_acquire = acquisition_function(relevant_dataset,save_path_dir,epoch_count,seed,metric,posterior_dict_new,modality_dict_new,gt_labels_dict_new,acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,classification,acquisition_percent=acquisition_percent,acquisition_metric_dict=acquisition_metric_dict,task_names_dict=task_names_dict_new,trial=trial) #,highest=highest)
                        #torch.save(acquisition_metric_dict,'acquisition_metric_dict')
                else:
                    retrieval_buffer_dict = storage_buffer_dict
                    samples_to_acquire = {task_name:len(elements) for task_name,elements in storage_buffer_dict.items()}
            
            """ At the Phase Level """
            if 'train1' in phase or 'val' in phase or inference == False:
                try:
                    print('%s Loss: %.4f. Acc: %.4f. AUROC: %.4f' % (phase,epoch_loss,epoch_acc,epoch_auroc))
                except:
                    print('%s Acc: %.4f. AUROC: %.4f' % (phase,epoch_acc,epoch_auroc))
                #print(scoring_function)
                #print('val_%s_%s_%s' % (downstream_dataset,modalities[0],leads),phase)
                
                #if task_instance_importance == True:
                    #torch.save(task_instance_params_dict,os.path.join(save_path_dir,'task_instance_params_dict'))
                    #torch.save(tracked_instance_params_dict,os.path.join(save_path_dir,'tracked_instance_params_dict'))
                    #print('Task Instance Params Saved!')
                if trial == 'multi_task_learning':
                    val_name = 'val'
                else:
                    val_name = 'val_%s_%s_%s_%s_%s' % (downstream_dataset,modalities[0],str(fraction),leads,class_pair)
                
                if val_name == phase and epoch_loss < best_loss:# or 'test' in phase and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = [copy.deepcopy(model.state_dict()) for model in models_list]
                    """ Save Best Finetuned Weights """
                    if 'train1' in phases:
                        save_config_weights(save_path_dir,best_model_wts)
                    
                    report, confusion = None, None
                    stop_counter = 0
                    print('Stop Counter %i' % stop_counter)
                elif val_name == phase and epoch_loss >= best_loss:
                    stop_counter += 1  
                    #print(stop_counter)
                
                #writer.add_scalar('%s_acc' % phase,epoch_acc,epoch_count)
                #writer.add_scalar('%s_loss' % phase,epoch_loss,epoch_count)
                #writer.add_scalar('%s_auc' % phase,epoch_auroc,epoch_count)
                acc_dict[phase].append(epoch_acc)
                loss_dict[phase].append(epoch_loss)
                auc_dict[phase].append(epoch_auroc)
                
        """ At the Epoch Level """
        if 'train1' in phases:
            prefix = 'train_val'
            save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict)
            if task_instance_importance == True:
                tracked_instance_params_dict,current_name = track_instance_params(epoch_count,task_instance_params_dict,tracked_instance_params_dict,current_task_info,new_task_epochs)
        
                if downstream_task == 'continual_buffer':
                    saving_epochs = list(np.array(list(new_task_dict.keys())[1:])-1)
                    if epoch_count in saving_epochs:
                        """ Perform Indices Retrieval Based on Some Metric for Buffer """
                        """ Current Name Encompasses Task, Modality, Leads, """
                        #previous_dataset = new_task_dict[epoch_count-np.diff(list(new_task_dict.keys()))[0]]
                        closest_epoch = epoch_count - epoch_count % np.diff(new_task_epochs)[0]
                        current_task_index = np.where([closest_epoch == epoch for epoch in new_task_epochs])[0][0]

                        buffer_indices = obtain_indices_for_buffer(current_task_index,tracked_loss,tracked_instance_params_dict,current_name,storage_percent,highest=highest)
                        storage_buffer_dict[current_name] = buffer_indices
                        torch.save(storage_buffer_dict,os.path.join(save_path_dir,'storage_buffer'))
                        """ What will Retrieval Buffer look like for transition Epochs """
                        #if epoch_count == list(new_task_dict.keys())[1]:
                        """ Adding All Indices From Most Recent Task At Transition Epochs """
                        retrieval_buffer_dict[current_name] = buffer_indices #only used for epochs in new_task_epochs
                        print(retrieval_buffer_dict)
                        print('Storage Buffer Saved!')
            elif task_instance_importance == False:
                if downstream_task == 'continual_buffer':
                    if trial == 'random_storage':
                        saving_epochs = list(np.array(list(new_task_dict.keys())[1:])-1)
                        if epoch_count in saving_epochs:
                            closest_epoch = epoch_count - epoch_count % np.diff(new_task_epochs)[0]
                            current_task_index = np.where([closest_epoch == epoch for epoch in new_task_epochs])[0][0]
                            nsamples_in_current_task = len(dataloaders_list[0]['train1'].batch_sampler.sampler.data_source.label_array)
                            """ Remove Number of Replayed Samples When Calculating Current Task NSamples """
                            if 'samples_to_acquire' in locals():
                                print(samples_to_acquire)
                                nreplayed_samples = np.sum(list(samples_to_acquire.values()))
                                nsamples_in_current_task -= nreplayed_samples
                            buffer_indices = obtain_random_storage_indices(current_task_index,storage_percent,nsamples_in_current_task)
                            storage_buffer_dict[current_name] = buffer_indices
                            torch.save(storage_buffer_dict,os.path.join(save_path_dir,'storage_buffer'))
                            retrieval_buffer_dict[current_name] = buffer_indices #only used for epochs in new_task_epochs
        else:
            break
        epoch_count += 1
    
    print('Best Val Loss: %.4f.' % best_loss)
    #output the model with best weights - to allow for forward pass outside
    print(phases)
    if 'train1' in phases:
        prefix = 'train_val'
        save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict)
        if task_instance_importance is True:
            torch.save(task_instance_params_dict,os.path.join(save_path_dir,'task_instance_params_dict'))
            torch.save(tracked_instance_params_dict,os.path.join(save_path_dir,'tracked_instance_params_dict'))
        [model.load_state_dict(best_model_wt) for model,best_model_wt in zip(models_list,best_model_wts)]
        print('Stats Saved!')
    elif 'val' in phases: #change name 
        prefix = 'val'
        #save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict)
    elif 'test' in phases:
        prefix = 'test'
        #save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict)        
    
    """ Obtain and Save Dicts to Evaluate Continual Learning Scenario """
    metric_name = ['acc','auc','loss']
    metric_list = [acc_dict,auc_dict,loss_dict]
    metric_dict = dict(zip(metric_name,metric_list))
    #print(acc_dict)
    ave_value_dict = dict()
    ave_bwt_dict = dict()
    ave_tstep_dict = dict()
    ave_lambda_dict = dict()
    for metric_name,metric in metric_dict.items():
        print(metric_name)
        ave_value = obtain_martha_acc(metric)
        ave_value_dict[metric_name] = ave_value
        
        if trial != 'multi_task_learning':
            ave_bwt = obtain_martha_bwt(metric,new_task_epochs)
            ave_bwt_dict[metric_name] = ave_bwt
            
            ave_tstep_bwt = obtain_tstep_bwt(metric,new_task_epochs)
            ave_tstep_dict[metric_name] = ave_tstep_bwt
            
            ave_lambda_bwt = obtain_lambda_bwt(metric,new_task_epochs)
            ave_lambda_dict[metric_name] = ave_lambda_bwt
    
    #metric_of_interest_to_print = 'auc'
    #print('Value: %.4f. BWT: %.4f. tStep-BWT: %.4f. Lambda-BWT: %.4f' % (ave_value_dict[metric_of_interest_to_print],ave_bwt_dict[metric_of_interest_to_print],ave_tstep_dict[metric_of_interest_to_print],ave_lambda_dict[metric_of_interest_to_print]))
    
    if trial == 'multi_task_learning':
        ave_dicts_names = ['value']
        ave_dicts_values = [ave_value_dict]
    else:
        ave_dicts_names = ['value','bwt','tstep_bwt','lambda_bwt'] 
        ave_dicts_values = [ave_value_dict,ave_bwt_dict,ave_tstep_dict,ave_lambda_dict]
        
    ave_dicts = dict(zip(ave_dicts_names,ave_dicts_values))
    save_continual_stats(save_path_dir,ave_dicts)
    
    return models_list, report, confusion, epoch_loss, epoch_auroc

def make_saving_directory_continual(trial,phases,downstream_task,cl_scenario,cl_strategy,heads,dataset_name,order,task_instance_importance,acquisition_epochs,storage_percent,acquisition_percent,seed,max_seed,highest,dropout_samples,acquisition_func):
    base_path = '/mnt/SecondaryHDD/Continual Learning Results' 
    order_path = 'order%s' % str(order)
    seed_path = 'seed%i' % int(seed)
    strategy_path = cl_strategy
    
    if cl_scenario == 'Domain-IL':
        dataset_path = dataset_name
    elif cl_scenario == 'Class-IL':
        dataset_path = dataset_name
    elif cl_scenario == 'Task-IL':
        dataset_path = dataset_name
    elif cl_scenario == 'Time-IL':
        dataset_path = dataset_name
    #else:
    #    dataset_path = ''
        
    heads_path = 'heads_%s' % heads
    
    if task_instance_importance == True:
        if downstream_task == 'continual_buffer':
            if trial == 'random_acquisition': #Random Acquisition Path
                task_instance_path = 'random_acquisition'
            else:
                task_instance_path = 'task_instance'
    elif task_instance_importance == False:
        if downstream_task == 'continual_buffer': 
            if trial == 'random_storage': #Random Storage Path
                task_instance_path = 'random_storage'
        elif downstream_task == 'multi_task_learning':
            task_instance_path = 'multi_task_learning'
        else: #Fine-Tuning Path e.g. downstream_task = 'los'
            task_instance_path = ''
    
    if len(acquisition_epochs) > 0:
        replay_path = 'replay'
        storage_path = 'storage_%.2f' % storage_percent[0]
        acquisition_path = 'acquisition_%.2f' % acquisition_percent[0]
        if highest == True:
            highest_path = ''
        else:
            highest_path = 'lowest'
        
        if 'monte_carlo' in trial: #use this if chain for extra ablation experiments if needed
            extra_experiment_path = 'monte_carlo'
            mc_path = str(dropout_samples)
        elif 'acquisition_function' in trial: #ablation to see effect of acquisition function chosen
            extra_experiment_path = 'acquisition_function'
            mc_path = acquisition_func
        else:
            extra_experiment_path = ''
            mc_path = ''
    else:
        replay_path = ''
        storage_path = ''
        acquisition_path = ''
        highest_path = ''
        extra_experiment_path = ''
        mc_path = ''
    
    save_path_dir = os.path.join(base_path,cl_scenario,heads_path,dataset_path,order_path,strategy_path,task_instance_path,replay_path,storage_path,acquisition_path,highest_path,extra_experiment_path,mc_path,seed_path)

    if 'train' in phases:
        save_path_dir, seed = make_dir(save_path_dir,max_seed) #base_path,order_path,seed_path,strategy_path,dataset_path,heads_path,task_instance_path,replay_path,storage_path,acquisition_path,seed_path,seed,max_seed)
    elif 'test' in phases:
        if 'test_auc' in os.listdir(save_path_dir):
            save_path_dir = 'do not test'
    
    return save_path_dir

def make_dir(save_path_dir,max_seed): #base_path,order_path,seed_path,strategy_path,dataset_path,heads_path,task_instance_path,replay_path,storage_path,acquisition_path,seed_path,seed,max_seed):
    """ Recursive Function to Make Sure I do Not Overwrite Previous Seeds """
    seed = int(save_path_dir.split('/')[-1].split('seed')[1])
    try:
        os.chdir(save_path_dir)
        if 'value' in os.listdir():
            if int(seed) < max_seed-1:
                print('Skipping Seed!')
                seed = int(seed) + 1
                seed_path = 'seed%i' % seed
                save_path_prefix = '/'.join(save_path_dir.split('/')[:-1])
                save_path_dir = os.path.join(save_path_prefix,seed_path)
                print(save_path_dir)
                save_path_dir, seed = make_dir(save_path_dir,max_seed) #base_path,extra_path,leads,trial,seed,max_seed,hyperparam,tolerance_path,noise_type_path,noise_level_path)
            else:
                save_path_dir = 'do not train'
    except:
        os.makedirs(save_path_dir)
    
    if int(seed) == max_seed:
        seed = 0
    
    return save_path_dir, int(seed)
