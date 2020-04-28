#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:31:49 2020

@author: Dani Kiyasseh
"""
#%%
""" Functions in this script:
    1) load_inputs_and_outputs
    2) data_transformations
    3) load_data_and_indices
    4) load_initial_data
    5) load_dataloaders_list_continual
    6) obtain_preceding_information
    7) obtain_dataloaders_information
    8) determine_label_offset_per_dataset
"""
#%%

import torch
from torch.utils.data import DataLoader
from prepare_dataset import my_dataset_direct
import numpy as np
import random
import os
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from cutout import Cutout
import pickle

from my_dataset_load_images import my_dataset
from prepare_miscellaneous import determine_classification_setting
#%%
def load_inputs_and_outputs(basepath,dataset_name,leads='i',cl_scenario=None):
    
    if dataset_name == 'bidmc':
        path = os.path.join(basepath,'BIDMC v1')
        extension = 'heartpy_'
    elif dataset_name == 'physionet':
        path = os.path.join(basepath,'PhysioNet v2')
        extension = 'heartpy_'
    elif dataset_name == 'mimic':
        shrink_factor = str(0.1)
        path = os.path.join(basepath,'MIMIC3_WFDB','frame-level',shrink_factor)
        extension = 'heartpy_'
    elif dataset_name == 'cipa':
        lead = ['II','aVR']
        path = os.path.join(basepath,'cipa-ecg-validation-study-1.0.0','leads_%s' % lead)
        extension = ''
    elif dataset_name == 'cardiology':
        classes = 'all'
        path = os.path.join(basepath,'CARDIOL_MAY_2017','patient_data','%s_classes' % classes)
        extension = ''
    elif dataset_name == 'physionet2017':
        path = os.path.join(basepath,'PhysioNet 2017','patient_data')
        extension = ''
    elif dataset_name == 'tetanus':
        path = '/media/scro3517/TertiaryHDD/new_tetanus_data/patient_data'
        extension = ''
    elif dataset_name == 'ptb':
        leads = [leads]
        path = os.path.join(basepath,'ptb-diagnostic-ecg-database-1.0.0','patient_data','leads_%s' % leads)
        extension = ''  
    elif dataset_name == 'fetal':
        abdomen = leads #'Abdomen_1'
        path = os.path.join(basepath,'non-invasive-fetal-ecg-arrhythmia-database-1.0.0','patient_data',abdomen)
        extension = ''
    elif dataset_name == 'physionet2016':
        path = os.path.join(basepath,'classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0')
        extension = ''
    elif dataset_name == 'physionet2020':
        #basepath = '/mnt/SecondaryHDD'
        leads = [leads]
        path = os.path.join(basepath,'PhysioNetChallenge2020_Training_CPSC','Training_WFDB','patient_data','leads_%s' % leads)
        extension = ''
    elif dataset_name == 'chapman':
        #basepath = '/mnt/SecondaryHDD'
        leads = leads
        path = os.path.join(basepath,'chapman_ecg','leads_%s' % leads)
        extension = ''
    elif dataset_name == 'cifar10':
        #basepath = '/mnt/SecondaryHDD'
        leads = ''
        path = os.path.join(basepath,'cifar-10-python/cifar-10-batches-py')
        extension = '' 

    if cl_scenario == 'Class-IL':
        dataset_name = dataset_name + '_' + 'mutually_exclusive_classes'

    """ Dict Containing Actual Frames """
    with open(os.path.join(path,'frames_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as f:
        input_array = pickle.load(f)
    """ Dict Containing Actual Labels """
    with open(os.path.join(path,'labels_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as g:
        output_array = pickle.load(g)
    
    return input_array,output_array,path

def data_transformations(operations,input_size):
    data_transforms = {
            'train': transforms.Compose([]),
            #transforms.Resize(input_size),
            #transforms.ToTensor()]),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            #transforms.Normalize([0.185, 0.521, 0.530], [0.182, 0.220, 0.148])]), #Ocean&Viridis FusionCol Training Data Batch 64
            #transforms.Normalize([0.254, 0.640, 0.466], [0.153, 0.140, 0.100])]), #Viridis FusionCol Batch 64
            'val': transforms.Compose([
            transforms.Resize(input_size), #resize input
            transforms.CenterCrop(input_size), #crop input 
            transforms.ToTensor()]), #convert input into tensor
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), #scale channel wise [means] [stds]
            #transforms.Normalize([0.185, 0.521, 0.530], [0.182, 0.220, 0.148])]),
            #transforms.Normalize([0.254, 0.640, 0.466], [0.153, 0.140, 0.100])]), 
            'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()]),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            #transforms.Normalize([0.185, 0.521, 0.530], [0.182, 0.220, 0.148])]),
            #transforms.Normalize([0.254, 0.640, 0.466], [0.153, 0.140, 0.100])]), 
            }
    """ Transforms to the train set """
    resize = operations['resize']
    affine = operations['affine']
    rotation = operations['rotation']
    color = operations['color']
    cutout = operations['perform_cutout']
    if resize is not False:
        print('Resize: %s' % str(resize))
        lower,upper = resize[0],resize[1]
        op = transforms.RandomResizedCrop(input_size,scale=(lower,upper),ratio=(lower,upper))
        data_transforms['train'].transforms.append(op)
    if affine is not False:
        print('Affine: %s' % str(affine))
        lower_degree,upper_degree = affine[0],affine[1]
        lower_scale,upper_scale = affine[2],affine[3]
        op = transforms.RandomAffine(degrees=[lower_degree,upper_degree],scale=(lower_scale,upper_scale),shear=[lower_degree,upper_degree])
        data_transforms['train'].transforms.append(op)
    if rotation:
        print('Rotation: %s' % str(rotation))
        op = transforms.RandomRotation(degrees=2)
        data_transforms['train'].transforms.append(op)
    if color:
        print('Color: %s' % str(color))
        brightness,contrast = color[0],color[1]
        op = transforms.ColorJitter(brightness=brightness,contrast=contrast,saturation=0,hue=0)
        data_transforms['train'].transforms.append(op)
    data_transforms['train'].transforms.append(transforms.Resize(input_size))
    data_transforms['train'].transforms.append(transforms.ToTensor())#final necessary elements of train transform
    if cutout is not False:
        print('Cutout: %s' % str(cutout))
        n_holes,length = cutout[0],cutout[1]
        data_transforms['train'].transforms.append(Cutout(n_holes=n_holes,length=length))    
    return data_transforms

def load_data_and_indices(dataname,code,classification='3-way'):
    enc = LabelEncoder()
    if 'physionet' in dataname:
        os.chdir('/home/scro3517/Desktop/PhysioNet 2015')
        X = np.load('x_physionet_%i.npy' % code)
        Y = np.load('y_physionet_%i.npy' % code)
        patient_numbers = np.load('patient_number_physionet_%i.npy' % code) 
        #leave 3 patients out (from each held-out set)
        class_patients = [np.unique(patient_numbers[Y==i]) for i in range(3)]
        random.seed(1)
        held_out_patients = [random.sample(list(class_patients[i]),2) for i in range(3)]
        validation_patients = [num[0] for num in held_out_patients]
        test_patients = [num[1] for num in held_out_patients]
        held_out_patients = [subnum for num in held_out_patients for subnum in num]
        #training, validation, and testing indices
        train_indices = np.where(np.in1d(patient_numbers,held_out_patients,invert=True))[0]
        validation_indices = np.where(np.in1d(patient_numbers,validation_patients))[0]
        test_indices = np.where(np.in1d(patient_numbers,test_patients))[0]
        #list of indices for later 
        phases = ['train','val','test']
        indices = [train_indices,validation_indices,test_indices]
        indices = dict(zip(phases,indices))
    elif 'fake' in dataname:
        os.chdir('/home/scro3517/Desktop/TSRTR')
        X = np.load('xfake_combo_%i_18K.npy' % code)
        Y = np.array([i for i in range(3) for _ in range(X.shape[0]//6)])
        Y = np.concatenate((Y,Y))
        #reshape data b/c we are working with 30-second frames 
        X = X.reshape(X.shape[0]//6,X.shape[1]*6)
        Y = Y[np.arange(0,Y.shape[0],6)]
        #train/val split in case I want to use MAML
        train_amount = int(0.8*X.shape[0])
        shuffle_indices = random.sample(range(X.shape[0]),X.shape[0])
        train_indices = shuffle_indices[:train_amount]
        val_indices = shuffle_indices[train_amount:]   
        phases = ['train','val']
        indices = [train_indices,val_indices]    
        indices = dict(zip(phases,indices))
    elif 'hfm' in dataname and 'ppg' in dataname:
        os.chdir('/home/scro3517/Desktop/HFM PPG')
        seconds = 30
        overlap = 0.95
        X = np.load('Frames/frames_%i_seconds_%.2f_overlap.npy' % (seconds,overlap),mmap_mode='r')
        Y = np.load('Labels/labels_%i_seconds_%.2f_overlap.npy' % (seconds,overlap))
        patient_numbers = np.load('Patient Numbers/patient_numbers_%i_seconds_%.2f_overlap.npy' % (seconds,overlap))
        """ Patient Numbers for Each Specific Class """
        if classification == '3-way':
            class_numbers = [0,1,2]
        elif classification == '2-way':
            class_numbers = [0,2]
            keep_patient_indices = np.where(np.in1d(Y,[class_numbers]))[0]
            X = X[keep_patient_indices,:]
            Y = list(itemgetter(*keep_patient_indices)(Y))
            Y = enc.fit_transform(Y)
            patient_numbers = list(itemgetter(*keep_patient_indices)(patient_numbers))
        nclasses = int(classification.split('-')[0])
        #class_patients = [patient_numbers[Y==i] for i in range(nclasses)]
        class_patients = [list(itemgetter(*np.where(Y==i)[0])(patient_numbers)) for i in range(nclasses)]
        """ Unique Patient Numbers for Each Specific Class """
        npatients_per_class = 3
        unique_patient_numbers = [np.unique(numbers) for numbers in class_patients]
        random.seed(0)
        held_out_patients = [random.sample(list(class_numbers),npatients_per_class*2) for class_numbers in unique_patient_numbers]
        train_indices = np.where(np.in1d(patient_numbers,held_out_patients,invert=True))[0]
        """ List of Val and Test Patient Numbers """
        val_patient_numbers = np.concatenate([held_out[:npatients_per_class] for held_out in held_out_patients])
        test_patient_numbers = np.concatenate([held_out[npatients_per_class:] for held_out in held_out_patients])
        """ Indices of Val and Test Patients """
        val_indices = np.where(np.in1d(patient_numbers,val_patient_numbers))[0]
        test_indices = np.where(np.in1d(patient_numbers,test_patient_numbers))[0]
        phases = ['train','val','test']
        indices = [train_indices,val_indices,test_indices]
        indices = dict(zip(phases,indices))
        
    data = {'inputs':X,'outputs':Y,'indices':indices}

    return data

def load_initial_data(basepath_to_data,phases,classification,fraction,inferences,unlabelled_fraction,labelled_fraction,test_representation,test_order,test_colourmap,test_dim,test_task,batch_size,modality,acquired_indices,acquired_labels,downstream_task,modalities,dataset_name,leads='ii',mixture='independent'):    
    """ Control augmentation at beginning of training here """ 
    resize = False
    affine = False
    rotation = False
    color = False    
    perform_cutout = False
    operations = {'resize': resize, 'affine': affine, 'rotation': rotation, 'color': color, 'perform_cutout': perform_cutout}    
    shuffles = {'train1':True,
                'train2':False,
                'val': False,
                'test': False}
    
    #data_transforms = data_transformations(operations,test_dim)
    
    """ Just Commented Out - December 17 2019 """
    #torch.manual_seed(0) #needed before each dataloader to ensure each dataset is alligned in the mixture case 
    #torch.cuda.manual_seed(0)
    
    #data_dirs = ['/home/scro3517/Desktop/TSRTR/%s/%s/%s' % (datatype,test_color,train_folder) for datatype,test_color in zip(datatypes_list,test_colors_list)] #combined Ocean and Viridis together
    #dataset_list = [{x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in phases} for data_dir in data_dirs] 
    
    
#    """ Dataloader - Image-Based """
#    task_data = {test_task:load_data_and_indices(test_task,0,classification)}
#    #print(task_data)
#    dataset_list = [{phase: my_dataset(test_task,task_data[test_task],phase,test_representation,test_order,test_colourmap,test_dim,data_transforms[phase],modality) for phase in phases}]
#    """ Dataloader - Image-Based """
    fractions = {'fraction': fraction,
                 'labelled_fraction': labelled_fraction,
                 'unlabelled_fraction': unlabelled_fraction}
    
    acquired_items = {'acquired_indices': acquired_indices,
                      'acquired_labels': acquired_labels}
    
    dataset_list = [{phase:my_dataset_direct(basepath_to_data,dataset_name,phase,inference,fractions,acquired_items,modalities=modalities,task=downstream_task,leads=leads) for phase,inference in zip(phases,inferences)}]                                        
    
    if 'train' in phases:
        check_dataset_allignment(mixture,dataset_list)
        
    dataloaders_list = [{phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=shuffles[phase],drop_last=False) for phase in phases} for dataset in dataset_list]
    print(len(dataloaders_list))
    
    return dataloaders_list,operations

""" Use this for Continual Learning """
def load_dataloaders_list_continual(basepath_to_data,fractions_list,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities_list,downstream_task,relevant_datasets,leads_list,cl_scenario,storage_buffer_dict=None,retrieval_buffer_dict=None,noutputs=None,input_perturbed=False,heads='multi',class_pairs_list=None):   
    shuffles = {'train1':True,
                'train2':False,
                'val': False,
                'test': False}
        
    fractions = {phase: {'fraction': fraction,
                         'labelled_fraction': labelled_fraction,
                         'unlabelled_fraction': unlabelled_fraction} for phase,fraction in zip(phases,fractions_list)}
    
    acquired_items = {'acquired_indices': acquired_indices,
                      'acquired_labels': acquired_labels,
                      'storage_buffered_indices': storage_buffer_dict,
                      'retrieval_buffered_indices': retrieval_buffer_dict,
                      'noutputs': noutputs}
    
    #if dataset_name == 'mimic_all': #when working indirectly with frames
    #    dataset_list = [{phase:my_dataset_indirect(dataset_name,phase,fraction,inference,unlabelled_fraction,labelled_fraction,acquired_indices=acquired_indices,acquired_predictions_dict=acquired_labels,task=downstream_task) for phase,inference in zip(phases,inferences)}]
    #else: #when directly working with frames 
        #print(phases,inferences)
    #print(relevant_datasets,phases,inferences,fractions,downstream_task,leads_list)
    dataset_list = [{phase:my_dataset_direct(basepath_to_data,dataset_name,phase,inference,fractions[phase],acquired_items,modalities=modalities,task=downstream_task,input_perturbed=input_perturbed,leads=leads,heads=heads,cl_scenario=cl_scenario,class_pair=class_pair) for dataset_name,phase,inference,modalities,leads,class_pair in zip(relevant_datasets,phases,inferences,modalities_list,leads_list,class_pairs_list)}]                                        
    
    check_dataset_allignment(mixture,dataset_list)
    #print('Batchsize: %i' % batch_size)
    #print('Active Dataloaders!')
    shuffles = {phase:shuffles[phase.split('_')[0]] for phase in phases} #adapt shuffles to new phase names
    #print(shuffles,phases)
    dataloaders_list = [{phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=shuffles[phase],drop_last=False) for phase in phases} for dataset in dataset_list]
    
    return dataloaders_list

def load_dataloaders_list(basepath_to_data,epoch_count,classification,fraction,inferences,unlabelled_fraction,labelled_subsample_fraction,mixture,test_representation,test_order,test_colourmap,test_dim,test_task,dataloaders_list,batch_size,modality,weighted_sampling,scoring_function,operations,downstream_task,dataset_name='mimic'):    
    """ Load Dataloaders Mid-Training for Augmentation Purposes """       
    if epoch_count != 0:
        """ Change transition epochs to dictate process """
        transition_epochs = None #epochs at which change occurs | None | [17]
        if transition_epochs is not None:
            """ Applying Augmentation at Pre-defined Intervals (2) """
            if epoch_count == transition_epochs[0]: #apply augmentation from this epoch onwards
                resize = [0.98,1.02] #this alone gets me to 0.87ish [0.98,1.02]
                affine = False #[-2,2,0.98,1.02] 
                rotation = False #[5]
                color = [0.2,0.2] #[0.2,0.2]
                perform_cutout = False
    
            if epoch_count in transition_epochs: #prevents repitition of the block (2)
                operations = {'resize': resize, 'affine': affine, 'rotation': rotation, 'color': color, 'perform_cutout': perform_cutout}    
                data_transforms = data_transformations(operations,test_dim)
                torch.manual_seed(0) #needed before each dataloader to ensure each dataset is alligned in the mixture case
                torch.cuda.manual_seed(0) #I dont think these do anything at this point of script
                #data_dirs = ['/home/scro3517/Desktop/TSRTR/%s/%s/%s' % (datatype,test_color,train_folder) for datatype,test_color in zip(datatypes_list,test_colors_list)] #combined Ocean and Viridis together
                #dataset_list = [{x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in phases} for data_dir in data_dirs] 
                """ Dataloader - Image-Based """
                #codes = [name.split('e')[-1] if 'fake' in name else name.split('net')[-1] for name in tasks]
                phases = ['train','val','test']
                task_data = {test_task:load_data_and_indices(test_task,0,classification)}
                dataset_list = [{phase:my_dataset(basepath_to_data,task_data[test_task],phase,test_representation,test_order,test_colourmap,test_dim,data_transforms[phase],modality) for phase in phases}]
                """ Dataloader - Image-Based """
                
                dataset_list = [{phase:my_dataset_direct(dataset_name,phase) for phase in phases}]                                        
                
                check_dataset_allignment(mixture,dataset_list)
                print('Batchsize: %i' % batch_size)
                dataloaders_list = [{phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=True,drop_last=True) for phase in phases} for dataset in dataset_list]
            else: #ensures we do not unnecessarily load datasets each epoch
                dataloaders_list = dataloaders_list
        
        if weighted_sampling:
            #sampling based on the loss experienced by each training input
            """ Easiest to Hardest Samples """
            scoring_function = 1/scoring_function
            """ Hardest to Easiest Samples """
            #scoring_function = scoring_function
            data_transforms = data_transformations(operations,test_dim)
            #torch.manual_seed(0) #needed before each dataloader to ensure each dataset is alligned in the mixture case
            #torch.cuda.manual_seed(0)
            
            """ Removed """
            #data_dirs = ['/home/scro3517/Desktop/TSRTR/%s/%s/%s' % (datatype,test_color,train_folder) for datatype,test_color in zip(datatypes_list,test_colors_list)] #combined Ocean and Viridis together
            #dataset_list = [{x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in phases} for data_dir in data_dirs] 
            #check_dataset_allignment(mixture,dataset_list)
            
            samplers = {}
            samplers['train'] = torch.utils.data.sampler.WeightedRandomSampler(scoring_function,num_samples=len(scoring_function),replacement=False)
            samplers['val'] = torch.utils.data.sampler.RandomSampler(dataset_list[0]['val']) #order doesn't matter for validation or test
            
            #dataloaders_list = [{x:DataLoader(dataset[x],batch_size=batch_size,shuffle=False,sampler=samplers[x],drop_last=True) for x in phases} for dataset in dataset_list]
    
    return dataloaders_list,operations

def obtain_preceding_information(epoch_count,new_task_epochs,cl_scenario,heads,new_task_info,trial):
    new_task_datasets, new_task_modalities, new_task_leads, new_task_class_pairs, new_task_fractions = new_task_info['new_task_datasets'], new_task_info['new_task_modalities'], new_task_info['new_task_leads'], new_task_info['new_task_class_pairs'], new_task_info['new_task_fractions'] 
    closest_epoch = epoch_count - epoch_count % np.diff(new_task_epochs)[0]
    current_task_index = np.where([closest_epoch == epoch for epoch in new_task_epochs])[0][0]
    #current_task_index = np.where([downstream_dataset in dataset for dataset in new_task_datasets])[0][0] #return index of current dataset
    #print('Current Task Index: %i' % current_task_index)
    preceding_datasets = new_task_datasets[:current_task_index] #datasets before current one
    
    extra_phases = []
    preceding_modalities = []
    preceding_leads = []
    preceding_class_pairs = []
    preceding_fractions = []
    for i,dataset in enumerate(preceding_datasets):
        preceding_modality = new_task_modalities[i]
        preceding_lead = new_task_leads[i]
        preceding_class_pair = new_task_class_pairs[i]
        preceding_fraction = new_task_fractions[i]
        
        preceding_modalities.append(preceding_modality) #modalities for preceding val datasets
        extra_phases.append('_'.join(('val',dataset,preceding_modality[0],str(preceding_fraction),preceding_lead,preceding_class_pair)))
        preceding_leads.append(preceding_lead)
        preceding_class_pairs.append(preceding_class_pair)
        preceding_fractions.append(preceding_fraction)
    
    """ Output Neurons for Single-Head CL """
    classification_per_dataset = [determine_classification_setting(dataset,cl_scenario,trial) for dataset in new_task_datasets]
    classification_per_dataset = [1 if classification == '2-way' else int(classification.split('-')[0]) for classification in classification_per_dataset]
    if heads == 'single':
        offset_per_dataset = np.cumsum(classification_per_dataset)
        offset_per_dataset = [0] + list(offset_per_dataset[:-1]) #- offset_per_dataset[0] 
        if cl_scenario == 'Class-IL' or cl_scenario == 'Time-IL':
            offset_per_dataset = [0] * len(new_task_datasets)
    else:
        offset_per_dataset = [0] * len(new_task_datasets) #no offset for multiple heads
    dataset_and_offset = dict(zip(new_task_datasets,offset_per_dataset))
    #print(dataset_and_offset)
    return extra_phases, preceding_datasets, preceding_modalities, preceding_leads, preceding_class_pairs, preceding_fractions, dataset_and_offset

def determine_label_offset_per_dataset(new_task_datasets,cl_scenario,trial,heads):
    """ Args:
        new_task_dataset = list of strings containing dataset names e.g. physionet """
    classification_per_dataset = [determine_classification_setting(dataset,cl_scenario,trial) for dataset in new_task_datasets]
    classification_per_dataset = [1 if classification == '2-way' else int(classification.split('-')[0]) for classification in classification_per_dataset]
    if heads == 'single':
        offset_per_dataset = np.cumsum(classification_per_dataset)
        offset_per_dataset = [0] + list(offset_per_dataset[:-1]) #- offset_per_dataset[0] 
        if cl_scenario == 'Class-IL':
            offset_per_dataset = [0] * len(new_task_datasets)
    else:
        offset_per_dataset = [0] * len(new_task_datasets) #no offset for multiple heads
    dataset_and_offset = dict(zip(new_task_datasets,offset_per_dataset))
    return dataset_and_offset

def obtain_dataloaders_information(basepath_to_data,acquisition_epochs,sample_epochs,new_task_epochs,metric,epoch_count,input_perturbed,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,class_pair,downstream_task,downstream_dataset,dataloaders_list,relevant_datasets,leads,storage_buffer_dict,retrieval_buffer_dict,heads,cl_scenario,new_task_info,trial=''):
    """ Acquisition Epochs = when to perform MC forward passes on storage buffer
        Sample Epochs = when to sample from retrieval buffer """
    #print(len(acquisition_epochs),trial)
    if len(acquisition_epochs) == 0:
        if trial == 'multi_task_learning':
            if epoch_count == 0: #you only need to load this once at beginning of training. 
                nphases = len(phases)
                relevant_datasets = [downstream_dataset]*nphases #2
                fractions_list = [fraction]*nphases #list of lists
                class_pairs_list = [class_pair]*nphases #list of lists
                modalities_list = [modalities]*nphases
                leads_list = [leads]*nphases
                dataset_and_offset = determine_label_offset_per_dataset(downstream_dataset,cl_scenario,trial,heads)
                print(relevant_datasets,phases,inferences,modalities_list,leads_list,class_pairs_list,dataset_and_offset)
                dataloaders_list = load_dataloaders_list_continual(basepath_to_data,fractions_list,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities_list,downstream_task,relevant_datasets,leads_list,cl_scenario,heads=heads,class_pairs_list=class_pairs_list,noutputs=dataset_and_offset)
                perturbed_dataloaders_list = None            
        else:
            if epoch_count in new_task_epochs or epoch_count in sample_epochs:# and epoch_count == 0: #for normal training path - nothing funky 
                extra_phases, preceding_datasets, preceding_modalities, preceding_leads, preceding_class_pairs, preceding_fractions, dataset_and_offset = obtain_preceding_information(epoch_count,new_task_epochs,cl_scenario,heads,new_task_info,trial)
    #            closest_epoch = epoch_count - epoch_count % np.diff(new_task_epochs)[0]
    #            current_task_index = np.where([closest_epoch == epoch for epoch in new_task_epochs])[0][0]
    #            #current_task_index = np.where([downstream_dataset in dataset for dataset in new_task_datasets])[0][0] #return index of current dataset
    #            print('Current Task Index: %i' % current_task_index)
    #            preceding_datasets = new_task_datasets[:current_task_index] #datasets before current one
    #            
    #            extra_phases = []
    #            preceding_modalities = []
    #            preceding_leads = []
    #            for i,dataset in enumerate(preceding_datasets):
    #                preceding_modality = new_task_modalities[i]
    #                preceding_lead = new_task_leads[i]
    #                
    #                preceding_modalities.append(preceding_modality) #modalities for preceding val datasets
    #                extra_phases.append('_'.join(('val',dataset,preceding_modality[0],preceding_lead)))
    #                preceding_leads.append(preceding_lead)
    #            
    #            """ Output Neurons for Single-Head CL """
    #            classification_per_dataset = [determine_classification_setting(dataset) for dataset in new_task_datasets]
    #            classification_per_dataset = [1 if classification == '2-way' else int(classification.split('-')[0]) for classification in classification_per_dataset]
    #            offset_per_dataset = np.cumsum(classification_per_dataset)
    #            offset_per_dataset = [0] + list(offset_per_dataset[:-1]) #- offset_per_dataset[0] 
    #            dataset_and_offset = dict(zip(new_task_datasets,offset_per_dataset))
                
                #print(sample_epochs)
                #print(epoch_count)
                if downstream_task == 'continual_buffer':
                    if epoch_count in sample_epochs: #when to sample and augment from buffer 
                        training_inference = True #True means sample from retrieval_buffer_dict later on #query for loading storage_buffer_dict
                else:
                    training_inference = False
                
                phases = ['train1','val_%s_%s_%s_%s_%s' % (downstream_dataset,modalities[0],str(fraction),leads,class_pair)] + extra_phases #add extra phases for preceding datasets
                inferences = [training_inference,False] + [False for _ in range(len(extra_phases))] #keep consistent with added phases 
                relevant_datasets = [downstream_dataset,downstream_dataset] + preceding_datasets #current dataset repeated twice, then preceding ones added
                modalities_list = [modalities,modalities] + preceding_modalities
                leads_list = [leads,leads] + preceding_leads
                class_pairs_list = [class_pair,class_pair] + preceding_class_pairs
                fractions_list = [fraction,fraction] + preceding_fractions
                #print(modalities_list)
                #print(relevant_datasets)
                #previous_datasets = #extend dataloaders list to get val data for the previous datasets (USE zip)
                """ Actual DataLoader """
                dataloaders_list = load_dataloaders_list_continual(basepath_to_data,fractions_list,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities_list,downstream_task,relevant_datasets,leads_list,cl_scenario,storage_buffer_dict=storage_buffer_dict,retrieval_buffer_dict=retrieval_buffer_dict,noutputs=dataset_and_offset,heads=heads,class_pairs_list=class_pairs_list)
                perturbed_dataloaders_list = None
                #print('New Dataset: %s' % downstream_dataset)
    elif len(acquisition_epochs) > 0:
        """ Epochs to Perform Acquisition At ---- Make Sure Sample Epochs Start > Acquisition Epochs Start > New Task Epochs[1] """
        if 'time' not in metric:
            if epoch_count in new_task_epochs[1:]:
                """ Perform Acquisition and Forward Pass with Augmented Set On Transition Epochs Too! """
                extra_phases, preceding_datasets, preceding_modalities, preceding_leads, preceding_class_pairs, preceding_fractions, dataset_and_offset = obtain_preceding_information(epoch_count,new_task_epochs,cl_scenario,heads,new_task_info,trial)
                phases = ['train2','train1','val_%s_%s_%s_%s_%s' % (downstream_dataset,modalities[0],str(fraction),leads,class_pair)] + extra_phases
                inferences = ['query',True,False] + [False for _ in range(len(extra_phases))]
                relevant_datasets = [downstream_dataset,downstream_dataset,downstream_dataset] + preceding_datasets #current dataset repeated twice, then preceding ones added
                modalities_list = [modalities,modalities,modalities] + preceding_modalities
                leads_list = [leads,leads,leads] + preceding_leads           
                class_pairs_list = [class_pair,class_pair,class_pair] + preceding_class_pairs
                #fractions_for_buffer_loading = preceding_fractions + [fraction]
                fractions_list = [fraction,fraction,fraction] + preceding_fractions
            elif epoch_count in acquisition_epochs: #when to perform MC forward passes
                extra_phases, preceding_datasets, preceding_modalities, preceding_leads, preceding_class_pairs, preceding_fractions, dataset_and_offset = obtain_preceding_information(epoch_count,new_task_epochs,cl_scenario,heads,new_task_info,trial)
                phases = ['train1','val_%s_%s_%s_%s_%s' % (downstream_dataset,modalities[0],str(fraction),leads,class_pair),'train2'] + extra_phases
                
                if downstream_task == 'continual_buffer':
                    if epoch_count in sample_epochs: #when to sample and augment from buffer 
                        train1_inference = True #True means sample from retrieval_buffer_dict later on #query for loading storage_buffer_dict
                        #fractions_for_buffer_loading = preceding_fractions + [fraction]
                    else:
                        train1_inference = False
                else:
                    train1_inference = False
                
                inferences = [train1_inference,False,'query'] + [False for _ in range(len(extra_phases))]
                relevant_datasets = [downstream_dataset,downstream_dataset,downstream_dataset] + preceding_datasets #current dataset repeated twice, then preceding ones added
                modalities_list = [modalities,modalities,modalities] + preceding_modalities
                leads_list = [leads,leads,leads] + preceding_leads
                class_pairs_list = [class_pair,class_pair,class_pair] + preceding_class_pairs
                fractions_list = [fraction,fraction,fraction] + preceding_fractions

            elif epoch_count in sample_epochs:
                extra_phases, preceding_datasets, preceding_modalities, preceding_leads, preceding_class_pairs, preceding_fractions, dataset_and_offset = obtain_preceding_information(epoch_count,new_task_epochs,cl_scenario,heads,new_task_info,trial)
                phases = ['train1','val_%s_%s_%s_%s_%s' % (downstream_dataset,modalities[0],str(fraction),leads,class_pair)] + extra_phases
                inferences = [True,False] + [False for _ in range(len(extra_phases))]
                relevant_datasets = [downstream_dataset,downstream_dataset] + preceding_datasets #current dataset repeated twice, then preceding ones added
                modalities_list = [modalities,modalities] + preceding_modalities
                leads_list = [leads,leads] + preceding_leads
                class_pairs_list = [class_pair,class_pair] + preceding_class_pairs
                
                #fractions_for_buffer_loading = preceding_fractions + [fraction]
                fractions_list = [fraction,fraction] + preceding_fractions

#                dataloaders_list = load_dataloaders_list_continual(fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,downstream_task,downstream_dataset)
#                if input_perturbed == True:
#                    #""" This Seed Ensures Perturbation is Same Across MC Passes But Different For Different Epochs - CONFIRMED """
#                    #np.random.seed(epoch_count)
#                    """ For now - this is just a filler - less flexibility """
#                    perturbed_dataloaders_list = load_dataloaders_list_continual(fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,downstream_task,downstream_dataset,input_perturbed)
            else:
                """ Ensure No Inference is Performed for Other Epochs """
                extra_phases, preceding_datasets, preceding_modalities, preceding_leads, preceding_class_pairs, preceding_fractions, dataset_and_offset = obtain_preceding_information(epoch_count,new_task_epochs,cl_scenario,heads,new_task_info,trial)
                phases = ['train1','val_%s_%s_%s_%s_%s' % (downstream_dataset,modalities[0],str(fraction),leads,class_pair)] + extra_phases
                inferences = [False,False] + [False for _ in range(len(extra_phases))]
                relevant_datasets = [downstream_dataset,downstream_dataset] + preceding_datasets #current dataset repeated twice, then preceding ones added
                modalities_list = [modalities,modalities] + preceding_modalities
                leads_list = [leads,leads] + preceding_leads
                class_pairs_list = [class_pair,class_pair] + preceding_class_pairs
                fractions_list = [fraction,fraction] + preceding_fractions

            """ Actual DataLoader """
            #dataloaders_list = load_dataloaders_list_continual(fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,downstream_task,downstream_dataset)
            dataloaders_list = load_dataloaders_list_continual(basepath_to_data,fractions_list,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities_list,downstream_task,relevant_datasets,leads_list,cl_scenario,storage_buffer_dict=storage_buffer_dict,retrieval_buffer_dict=retrieval_buffer_dict,noutputs=dataset_and_offset,heads=heads,class_pairs_list=class_pairs_list)
            if input_perturbed == True:
                #perturbed_dataloaders_list = load_dataloaders_list_continual(fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,downstream_task,downstream_dataset,input_perturbed)
                perturbed_dataloaders_list = load_dataloaders_list_continual(basepath_to_data,fractions_list,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities_list,downstream_task,relevant_datasets,leads_list,cl_scenario,storage_buffer_dict=storage_buffer_dict,retrieval_buffer_dict=retrieval_buffer_dict,noutputs=dataset_and_offset,heads=heads,input_perturbed=input_perturbed,class_pairs_list=class_pairs_list)
        else:
            """ Time in Metric ==> MC on Every Epoch """
            extra_phases, preceding_datasets, preceding_modalities, preceding_leads, preceding_class_pairs, preceding_fractions, dataset_and_offset = obtain_preceding_information(epoch_count,new_task_epochs,cl_scenario,heads,new_task_info,trial)
            phases = ['train1','val_%s_%s_%s_%s_%s' % (downstream_dataset,modalities[0],str(fraction),leads,class_pair),'train2'] + extra_phases
            
            if downstream_task == 'continual_buffer':
                if epoch_count in sample_epochs: #when to sample and augment from buffer 
                    train1_inference = True #True means sample from retrieval_buffer_dict later on #query for loading storage_buffer_dict
                    #fractions_for_buffer_loading = preceding_fractions + [fraction]
            else:
                train1_inference = False
            
            inferences = [train1_inference,False,'query'] + [False for _ in range(len(extra_phases))]
            relevant_datasets = [downstream_dataset,downstream_dataset,downstream_dataset] + preceding_datasets #current dataset repeated twice, then preceding ones added
            modalities_list = [modalities,modalities,modalities] + preceding_modalities
            leads_list = [leads,leads,leads] + preceding_leads
            class_pairs_list = [class_pair,class_pair,class_pair] + preceding_class_pairs
            fractions_list = [fraction,fraction,fraction] + preceding_fractions

            """ Actual DataLoader """
            #dataloaders_list = load_dataloaders_list_continual(fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities,downstream_task,downstream_dataset)
            dataloaders_list = load_dataloaders_list_continual(basepath_to_data,fractions_list,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities_list,downstream_task,relevant_datasets,leads_list,cl_scenario,storage_buffer_dict=storage_buffer_dict,retrieval_buffer_dict=retrieval_buffer_dict,noutputs=dataset_and_offset,heads=heads,class_pairs_list=class_pairs_list)
            if input_perturbed == True:
                #np.random.seed(epoch_count)
                perturbed_dataloaders_list = load_dataloaders_list_continual(basepath_to_data,fractions_list,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,batch_size,phases,modalities_list,downstream_task,relevant_datasets,leads_list,cl_scenario,storage_buffer_dict=storage_buffer_dict,retrieval_buffer_dict=retrieval_buffer_dict,noutputs=dataset_and_offset,heads=heads,input_perturbed=input_perturbed,class_pairs_list=class_pairs_list)
    
    if input_perturbed == True:
        return relevant_datasets,phases,inferences,dataloaders_list,perturbed_dataloaders_list
    elif input_perturbed == False:
        return relevant_datasets,phases,inferences,dataloaders_list

def check_dataset_allignment(mixture,dataset_list):
    if mixture:
        length_prev = 0 #starter
        for i in range(len(dataset_list)):
            length_curr = len(dataset_list[i]['train'])
            if i != 0:
                if length_curr != length_prev:
                    print('Caution! Datasets are not alligned')
                    exit()
            length_prev = length_curr