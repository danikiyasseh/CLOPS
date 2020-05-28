

import torch
from torch.utils.data import Dataset
from operator import itemgetter
import os
import pickle
import numpy as np
#import os
import random
#from scipy.stats import entropy
#import pandas as pd

import torchvision.transforms as transforms
#%%

class my_dataset_direct(Dataset):
    """ Takes Arrays and Phase, and Returns Sample 
        i.e. use for BIDMC and PhysioNet Datasets 
    """
    
    def __init__(self,basepath_to_data,dataset_name,phase,inference,fractions,acquired_items,modalities=['ecg','ppg'],task='self-supervised',input_perturbed=False,leads='ii',heads='single',cl_scenario=None,class_pair=''):
        """ This Accounts for 'train1' and 'train2' Phases """
        if 'train' in phase:
            phase = 'train'
        elif 'val' in phase:
            phase = 'val'

        self.basepath = basepath_to_data
        self.task = task #continual_buffer, etc. 
        self.cl_scenario = cl_scenario
        if task != 'multi_task_learning':
            input_array,output_array = self.load_raw_inputs_and_outputs(dataset_name,leads)
            self.output_array = output_array #original output dict
        #print(output_array['ecg'][0.9]['train']['labelled'].shape)
        fraction = fractions['fraction'] #needs to be a list when dealing with 'query' or inference = True for CL scenario
        labelled_fraction = fractions['labelled_fraction']
        unlabelled_fraction = fractions['unlabelled_fraction']
        acquired_indices = acquired_items['acquired_indices']
        acquired_labels = acquired_items['acquired_labels']
        self.dataset_and_offset = acquired_items['noutputs']
        #print(len(acquired_indices.values()))
        """ Combine Modalities into 1 Array """
        frame_array = []
        label_array = []
        self.modalities = modalities
        self.dataset_name = dataset_name
        self.heads = heads
        self.acquired_items = acquired_items
        self.fraction = fraction
        self.leads = leads
        self.class_pair = class_pair
#        self.name = '-'.join((dataset_name,modalities[0],str(fraction),leads,class_pair)) #name for different tasks
        
        if task == 'self-supervised':
            for modality in modalities:
                if phase == 'train':
                    modality_input = np.concatenate(list(input_array[modality][fraction][phase].values()))
                    modality_output = np.concatenate(list(output_array[modality][fraction][phase].values()))
                else:
                    modality_input = input_array[modality][fraction][phase]
                    modality_output = output_array[modality][fraction][phase]          
                    #modality_input = input_array[modality][fraction][phase][train_key]
                    #modality_output = output_array[modality][fraction][phase][train_key]
    
                frame_array.append(modality_input)
                label_array.append(modality_output)
                
            self.input_array = np.concatenate(frame_array)
            self.label_array = [i for i in range(len(modalities)) for _ in range(modality_input.shape[0])]
        elif task == 'continual_buffer':
            self.name = '-'.join((dataset_name,modalities[0],str(fraction),leads,class_pair)) #name for different tasks
            if phase == 'train':
                if inference == False:
                    inputs,outputs = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,dataset_name=self.dataset_name)
                    t = np.where([self.dataset_name == key for key in self.dataset_and_offset.keys()])[0] #obtain task index
                    outputs = self.offset_outputs(dataset_name,outputs,t)
                    keep_indices = list(np.arange(inputs.shape[0]))
                    dataset_list = [self.name for _ in range(len(keep_indices))]
                elif inference == 'query': #loaded for MC Sampling 
                    print('Storage Buffer Loading!')
                    buffer_indices_dict = acquired_items['storage_buffered_indices']
                    inputs,outputs,task_indices,task_name = self.retrieve_buffered_data(buffer_indices_dict,fraction,labelled_fraction)
                    keep_indices = task_indices
                    dataset_list = task_name
                    print('NSamples for MC: %s' % str(inputs.shape))
                elif inference == True:
                    print('Retrieval Buffer Loaded!')
                    #print(fraction)
                    """ Inference is True Means Perform Epoch Training With Current + Buffered Data """
                    buffer_indices_dict = acquired_items['retrieval_buffered_indices']
                    inputs,outputs,dataset_list = self.expand_labelled_data_with_buffer(input_array,output_array,buffer_indices_dict,fraction,labelled_fraction)
                    keep_indices = list(np.arange(inputs.shape[0])) 
                    #""" CHECK """
                    #dataset_list = [dataset_name for _ in range(len(keep_indices))]
                    #dataset_list = list(buffer_indices_dict.keys())
                    #outputs = self.offset_outputs(dataset_name,outputs)
                    print('Nsamples in Expanded Dataset: %s' % str(inputs.shape))
            else:
                inputs,outputs = self.retrieve_val_data(input_array,output_array,phase,fraction)
                t = np.where([self.dataset_name == key for key in self.dataset_and_offset.keys()])[0] #obtain task index
                outputs = self.offset_outputs(dataset_name,outputs,t)
                keep_indices = list(np.arange(inputs.shape[0])) #filler
                dataset_list = [dataset_name for _ in range(len(keep_indices))] 

            modality_array = list(np.arange(inputs.shape[0])) #filler
            
            self.dataset_name = dataset_name
            self.dataset_list = dataset_list
            self.modality_array = modality_array
            self.remaining_indices = keep_indices            
            self.input_array = inputs
            self.label_array = outputs   
        elif task == 'multi_task_learning':
            if phase == 'train':
                inputs, outputs = self.retrieve_multi_task_train_data()
            else:
                inputs, outputs = self.retrieve_multi_task_val_data(phase)
            
            keep_indices = list(np.arange(inputs.shape[0])) #filler
            modality_array = list(np.arange(inputs.shape[0])) #filler
            dataset_list = ['All' for _ in range(len(keep_indices))] #filler
            self.dataset_name = dataset_name
            self.dataset_list = dataset_list
            self.input_array = inputs
            self.label_array = outputs
            self.modality_array = modality_array
            self.remaining_indices = keep_indices
            
        else: #normal training path
            self.name = '-'.join((dataset_name,modalities[0],str(fraction),leads,class_pair)) #name for different tasks
            if phase == 'train':
                if inference == False:
                    inputs,outputs = self.expand_labelled_data(input_array,output_array,fraction,labelled_fraction,unlabelled_fraction,acquired_indices,acquired_labels)
                    #outputs = self.offset_outputs(dataset_name,outputs)
                    """ Added May 19th 2020 """
                    t = np.where([self.dataset_name == key for key in self.dataset_and_offset.keys()])[0] #obtain task index
                    outputs = self.offset_outputs(dataset_name,outputs,t)
                    """ End """                    
                    keep_indices = list(np.arange(inputs.shape[0])) #filler
                    modality_array = list(np.arange(inputs.shape[0])) #filler
                elif inference == True: #==> when MC Dropout is Performed
                    inputs,outputs,modality_array,keep_indices = self.retrieve_modified_unlabelled_data(input_array,output_array,fraction,unlabelled_fraction,acquired_indices)
                    #outputs = self.offset_outputs(dataset_name,outputs)
                    
                    #print(keep_indices)
                    #if input_perturbed == True: #perturb for consistency acquisition metrics
                    #    inputs = self.perturb_inputs(inputs,dataset_name)
            else:
                inputs,outputs = self.retrieve_val_data(input_array,output_array,phase,fraction)
                #outputs = self.offset_outputs(dataset_name,outputs)
                """ Added May 19th 2020 """
                t = np.where([self.dataset_name == key for key in self.dataset_and_offset.keys()])[0] #obtain task index
                outputs = self.offset_outputs(dataset_name,outputs,t)
                """ End """
                keep_indices = list(np.arange(inputs.shape[0])) #filler
                modality_array = list(np.arange(inputs.shape[0])) #filler
            
            dataset_list = [self.name for _ in range(len(keep_indices))] #filler
            self.dataset_name = dataset_name
            self.dataset_list = dataset_list
            self.input_array = inputs
            self.label_array = outputs
            self.modality_array = modality_array
            self.remaining_indices = keep_indices
        
        self.input_perturbed = input_perturbed #boolean that determinens consistency approach
        self.phase = phase
#    def perturb_inputs(self,inputs,dataset_name):
#        #if dataset_name in ['cardiology','physionet2017']:
#        variance_factor = 100 
#        #else:
#        #    variance_factor = 0.001 #0.01 was initially used for physionet 2015 ppg experiments
#        #gauss_noise = np.random.multivariate_normal(np.zeros((inputs.shape[1])),variance_factor*np.eye(inputs.shape[1]))
#        """ Univariate Normal is Much Faster to Compute """
#        gauss_noise = np.random.normal(0,variance_factor,size=(2500))
#        """ These are Pre-Normalization Inputs, Therefore Noise Has to be Large Enough """
#        inputs = inputs + gauss_noise
#        return inputs

    def load_raw_inputs_and_outputs(self,dataset_name,leads='i'):
        """ Load Arrays Based on dataset_name """
        basepath = self.basepath
        
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
            leads = 'all' #flexibility to change later 
            path = os.path.join(basepath,'CARDIOL_MAY_2017','patient_data','%s_classes' % leads)
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
            #leads = [leads] #original implementation
            leads = leads
            if self.cl_scenario == 'Task-IL':
                subfolder = 'contrastive_ss'
                leads = leads
            else:
                subfolder = ''
                leads = [leads]
            path = os.path.join(basepath,'PhysioNetChallenge2020_Training_CPSC','Training_WFDB','patient_data',subfolder,'leads_%s' % leads) #'contrastive_ss',
            extension = ''
        elif dataset_name == 'chapman':
            leads = leads
            path = os.path.join(basepath,'chapman_ecg','contrastive_ss','leads_%s' % leads)
            extension = ''
        elif dataset_name == 'uci_emg':
            #basepath = '/mnt/SecondaryHDD'
            leads = ''
            path = os.path.join(basepath,'UCI EMG Dataset')
            extension = ''         
        elif dataset_name == 'covid19':
            #basepath = '/mnt/SecondaryHDD'
            leads = ''
            path = os.path.join(basepath,'CURIAL Project')
            extension = ''                     
        elif dataset_name == 'cifar10':
            #basepath = '/mnt/SecondaryHDD'
            leads = ''
            path = os.path.join(basepath,'cifar-10-python/cifar-10-batches-py')
            extension = ''    
        elif dataset_name == 'ptbxl':
            #basepath = '/mnt/SecondaryHDD'
            leads = leads
            code_of_interest = 'rhythm' #options: 'rhythm' | 'all' #tells you the classification formulation 
            path = os.path.join(basepath,'PTB-XL','patient_data','leads_%s' % leads,'classes_%s' % code_of_interest)
            extension = ''                                    
            if self.cl_scenario == 'Device-IL': #Device 1 then Device 2 scenario
                path = os.path.join(path,'continual')                             
        
        if self.cl_scenario == 'Class-IL':
            dataset_name = dataset_name + '_' + 'mutually_exclusive_classes'
        
        """ Dict Containing Actual Frames """
        with open(os.path.join(path,'frames_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as f:
            input_array = pickle.load(f)
        """ Dict Containing Actual Labels """
        with open(os.path.join(path,'labels_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as g:
            output_array = pickle.load(g)
        
        return input_array,output_array

    def offset_outputs(self,dataset_name,outputs,t=0): #t tells you which class pair you are on now (used rarely and only for MTL)
        """ Offset Label Position in case of Single Head """
        self.dataset_and_offset = self.acquired_items['noutputs']
        if self.heads == 'single':
            """ Changed March 17th 2020 """
            offset = self.dataset_and_offset[dataset_name] #self.dataset_name
            """ End """
            if dataset_name == 'physionet2020' or dataset_name == 'ptbxl': #multilabel situation 
                """ Option 1 - Expansion """
                #noutputs = outputs.shape[1] * 12 #9 classes and 12 leads
                #expanded_outputs = np.zeros((outputs.shape[0],noutputs))
                #expanded_outputs[:,offset:offset+9] = outputs
                #outputs = expanded_outputs
                """ Option 2 - No Expansion """
                outputs = outputs 
                if self.cl_scenario == 'Task-IL':
                    pre_padding = []
                    post_padding = []
                    for pseudo_t,(dataset_name,offset) in enumerate(self.dataset_and_offset.items()):
                        if pseudo_t < t:
                            if dataset_name == 'physionet2020':
                                pad = np.repeat(0,9)
                            elif dataset_name == 'ptbxl':
                                pad = np.repeat(0,12)
                            else:
                                pad = np.repeat(0,4) #4) #4 represents number of classes in iter dataset
                            pre_padding.extend(pad)
                        elif pseudo_t > t:
                            if dataset_name == 'physionet2020':
                                pad = np.repeat(0,9)
                            elif dataset_name == 'ptbxl':
                                pad = np.repeat(0,12)
                            else:
                                pad = np.repeat(0,4) #4) #4 represents number of classes in iter dataset
                            post_padding.extend(pad) 
                    pre_padding_array = np.tile(pre_padding,[outputs.shape[0],1])
                    post_padding_array = np.tile(post_padding,[outputs.shape[0],1])
                    outputs = np.hstack((pre_padding_array,outputs,post_padding_array)) #outputs should already be multi-hot vector
                    #print('offset_outputs_function')
                    #print(outputs)
            else: 
                if dataset_name == 'cardiology' and self.task == 'multi_task_learning':
                    outputs = outputs + 2*t
                elif dataset_name == 'chapman' and self.task == 'multi_task_learning':
                    outputs = outputs
                else:
                    outputs = outputs + offset #output represents actual labels
                #print(offset)
        return outputs

    def retrieve_buffered_data(self,buffer_indices_dict,fraction,labelled_fraction):
        input_buffer = []
        output_buffer = []
        task_indices_buffer = []
        dataset_buffer = []
        #print(fraction_list)
        #for fraction,(task_name,indices) in zip(fraction_list[:-1],buffer_indices_dict.items()):
        for t,(task_name,indices) in enumerate(buffer_indices_dict.items()):
            #name = '-'.join((task,modality,leads,str(fraction))) #dataset, modality, fraction, leads
            dataset = task_name.split('-')[0]
            fraction = float(task_name.split('-')[2])
            leads = task_name.split('-')[3]
            if self.cl_scenario == 'Class-IL':
                self.class_pair = '-'.join(task_name.split('-')[-2:]) #b/c e.g. '0-1' you need last two
            elif self.cl_scenario == 'Time-IL':
                self.class_pair = task_name.split('-')[-1] 
            elif self.cl_scenario == 'Task-IL' and 'chapman' in dataset: #chapman ecg as task in Task-IL setting
                self.class_pair = task_name.split('-')[-1] 
            elif self.cl_scenario == 'Device-IL' and 'ptbxl' in dataset: #chapman ecg as task in Task-IL setting
                self.class_pair = task_name.split('-')[-1]             
            input_array,output_array = self.load_raw_inputs_and_outputs(dataset,leads)
            input_array,output_array = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,dataset_name=dataset)
            """ Offset Applied to Each Dataset """
            if self.heads == 'single':#'continual_buffer':
                output_array = self.offset_outputs(dataset,output_array,t)
                #offset = self.dataset_and_offset[dataset]
                #output_array = output_array + offset
            current_input_buffer,current_output_buffer = input_array[indices,:], output_array[indices]
            input_buffer.append(current_input_buffer)
            output_buffer.append(current_output_buffer)
            task_indices_buffer.append(indices) #will go 1-10K, 1-10K, etc. not cumulative indices
            dataset_buffer.append([task_name for _ in range(len(indices))])
        #print(input_buffer)
        input_buffer = np.concatenate(input_buffer,axis=0)
        output_buffer = np.concatenate(output_buffer,axis=0)
        task_indices_buffer = np.concatenate(task_indices_buffer,axis=0)
        dataset_buffer = np.concatenate(dataset_buffer,axis=0)
        return input_buffer,output_buffer,task_indices_buffer,dataset_buffer
                
    def expand_labelled_data_with_buffer(self,input_array,output_array,buffer_indices_dict,fraction,labelled_fraction):
        """ function arguments are raw inputs and outputs """
        input_buffer,output_buffer,task_indices_buffer,dataset_buffer = self.retrieve_buffered_data(buffer_indices_dict,fraction,labelled_fraction)
        if self.cl_scenario == 'Class-IL':
            self.class_pair = '-'.join(self.name.split('-')[-2:]) #b/c e.g. '0-1' you need last two
        elif self.cl_scenario == 'Time-IL':
            self.class_pair = self.name.split('-')[-1]
        #elif self.cl_scenario == 'Device-IL' and 'ptbxl' in dataset: #chapman ecg as task in Task-IL setting
        #    self.class_pair = task_name.split('-')[-1]    
        input_array,output_array = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,dataset_name=self.dataset_name)
        dataset_list = [self.name for _ in range(input_array.shape[0])]
        #print(max(output_array))
        """ Offset Applied to Current Dataset """
        if self.heads == 'single':#'continual_buffer':
            t = np.where([self.dataset_name == key for key in self.dataset_and_offset.keys()])[0] #obtain task index
            output_array = self.offset_outputs(self.dataset_name,output_array,t)
            #offset = self.dataset_and_offset[self.dataset_name]
            #print('Offset')
            #print(offset)
            #output_array = output_array + offset
        print(input_array.shape,input_buffer.shape)
        input_array = np.concatenate((input_array,input_buffer),0)
        output_array = np.concatenate((output_array,output_buffer),0)
        dataset_list = np.concatenate((dataset_list,dataset_buffer),0)
        #print(input_array.shape)
        #print(max(output_array),max(output_buffer))
        return input_array,output_array,dataset_list
    
    def retrieve_val_data(self,input_array,output_array,phase,fraction,labelled_fraction=1):#,modalities=['ecg','ppg']):
        frame_array = []
        label_array = []
        
        if self.cl_scenario == 'Class-IL' or self.cl_scenario == 'Time-IL' or self.cl_scenario == 'Device-IL' or (self.cl_scenario == 'Task-IL' and self.dataset_name == 'chapman'):        
            for modality in self.modalities:
                modality_input = input_array[modality][fraction][phase][self.class_pair]
                modality_output = output_array[modality][fraction][phase][self.class_pair]
                frame_array.append(modality_input)
                label_array.append(modality_output)
        else:
            """ Obtain Modality-Combined Unlabelled Dataset """
            for modality in self.modalities:
                modality_input = input_array[modality][fraction][phase]
                modality_output = output_array[modality][fraction][phase]
                frame_array.append(modality_input)
                label_array.append(modality_output)        
        
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)            
        
        inputs,outputs,_ = self.shrink_data(inputs,outputs,labelled_fraction)
        
        return inputs,outputs             
    
    def shrink_data(self,inputs,outputs,fraction,modality_array=None):
        nframes_to_sample = int(inputs.shape[0]*fraction)
        random.seed(0)
        indices = random.sample(list(np.arange(inputs.shape[0])),nframes_to_sample)
        inputs = np.array(list(itemgetter(*indices)(inputs)))
        outputs = np.array(list(itemgetter(*indices)(outputs)))
        if modality_array is not None:
            modality_array = np.array(list(itemgetter(*indices)(modality_array)))
        return inputs,outputs,modality_array
    
    def remove_acquired_data(self,inputs,outputs,modality_array,acquired_indices):
        keep_indices = list(set(list(np.arange(inputs.shape[0]))) - set(acquired_indices))
        inputs = np.array(list(itemgetter(*keep_indices)(inputs)))
        outputs = np.array(list(itemgetter(*keep_indices)(outputs)))
        modality_array = np.array(list(itemgetter(*keep_indices)(modality_array)))
        return inputs,outputs,modality_array,keep_indices
    
    def retrieve_unlabelled_data(self,input_array,output_array,fraction,unlabelled_fraction):#,modalities=['ecg','ppg']):
        phase = 'train'
        frame_array = []
        label_array = []
        modality_array = []
        
        """ Obtain Modality-Combined Unlabelled Dataset """
        for modality in self.modalities:
            modality_input = input_array[modality][fraction][phase]['unlabelled']
            modality_output = output_array[modality][fraction][phase]['unlabelled']
            modality_name = [modality for _ in range(modality_input.shape[0])]
            frame_array.append(modality_input)
            label_array.append(modality_output)
            modality_array.append(modality_name)
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)   
        modality_array = np.concatenate(modality_array)         
        
        inputs,outputs,modality_array = self.shrink_data(inputs,outputs,unlabelled_fraction,modality_array)
        
        return inputs,outputs,modality_array
        
    ### This is function you want for MC Dropout Phase ###
    def retrieve_modified_unlabelled_data(self,input_array,output_array,fraction,unlabelled_fraction,acquired_indices):
        inputs,outputs,modality_array = self.retrieve_unlabelled_data(input_array,output_array,fraction,unlabelled_fraction)
        inputs,outputs,modality_array,keep_indices = self.remove_acquired_data(inputs,outputs,modality_array,acquired_indices)
        return inputs,outputs,modality_array,keep_indices

    def retrieve_labelled_data(self,input_array,output_array,fraction,labelled_fraction,dataset_name=''):#,modalities=['ecg','ppg']):
        phase = 'train'
        frame_array = []
        label_array = []

        if self.cl_scenario == 'Class-IL' or self.cl_scenario == 'Time-IL' or self.cl_scenario == 'Device-IL':
            header = self.class_pair
        elif self.cl_scenario == 'Task-IL' and dataset_name == 'chapman':
            header = self.class_pair
        else:
            header = 'labelled'

        """ Obtain Modality-Combined Labelled Dataset """
        for modality in self.modalities:
            modality_input = input_array[modality][fraction][phase][header]
            modality_output = output_array[modality][fraction][phase][header]
            frame_array.append(modality_input)
            label_array.append(modality_output)
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)
        
        inputs,outputs,_ = self.shrink_data(inputs,outputs,labelled_fraction)

        return inputs,outputs

    def acquire_unlabelled_samples(self,inputs,outputs,fraction,unlabelled_fraction,acquired_indices):
        inputs,outputs,modality_array = self.retrieve_unlabelled_data(inputs,outputs,fraction,unlabelled_fraction)
        if len(acquired_indices) > 1:
            inputs = np.array(list(itemgetter(*acquired_indices)(inputs)))
            outputs = np.array(list(itemgetter(*acquired_indices)(outputs)))
            modality_array = np.array(list(itemgetter(*acquired_indices)(modality_array)))
        elif len(acquired_indices) == 1:
            """ Dimensions Need to be Adressed to allow for Concatenation """
            inputs = np.expand_dims(np.array(inputs[acquired_indices[0],:]),1)
            outputs = np.expand_dims(np.array(outputs[acquired_indices[0]]),1)
            modality_array = np.expand_dims(np.array(modality_array[acquired_indices[0]]),1)
        return inputs,outputs,modality_array

    def retrieve_multi_task_train_data(self):
        """ Load All Required Tasks for Multi-Task Training Setting """
        all_class_pairs = self.class_pair
        all_modalities = self.modalities
        input_array = []
        output_array = []
        for t,(dataset_name,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
            current_input, current_output = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
            self.class_pair = class_pair
            self.modalities = all_modalities[t] #list(map(lambda x:x[0],all_modalities))
            current_input, current_output = self.retrieve_labelled_data(current_input,current_output,fraction,1,dataset_name=dataset_name)
            current_output = self.offset_outputs(dataset_name,current_output,t)
            print(current_output.shape)
            input_array.append(current_input)
            output_array.append(current_output)
        input_array = np.concatenate(input_array,axis=0)
        output_array = np.concatenate(output_array,axis=0)
        print(output_array.shape)
        print(np.max(output_array))
        return input_array,output_array

    def retrieve_multi_task_val_data(self,phase):
        """ Load All Required Tasks for Multi-Task Validation/Testing Setting """
        all_class_pairs = self.class_pair
        all_modalities = self.modalities
        input_array = []
        output_array = []
        for t,(dataset_name,modalities,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,all_modalities,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
            current_input, current_output = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
            self.class_pair = class_pair
            self.modalities = modalities
            current_input, current_output = self.retrieve_val_data(current_input,current_output,phase,fraction)#,labelled_fraction=1)
            current_output = self.offset_outputs(dataset_name,current_output,t)
            input_array.append(current_input)
            output_array.append(current_output)
        input_array = np.concatenate(input_array,axis=0)
        output_array = np.concatenate(output_array,axis=0)
        
        return input_array,output_array

    ### This is function you want for training ###
    def expand_labelled_data(self,input_array,output_array,fraction,labelled_fraction,unlabelled_fraction,acquired_indices,acquired_labels):
        inputs,outputs = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,self.dataset_name)
        #print(self.remaining_indices)
        #print('Acquired Indices!')
        #print(acquired_indices)
        """ If indices have been acquired, then use them. Otherwise, do not """
        #if isinstance(acquired_indices,list):
        #    condition = len(acquired_indices) > 0
        #elif isinstance(acquired_indices,dict):
        #    condition = len(acquired_indices) > 1
        """ Changed March 5, 2020 """
        #if len(acquired_indices) > 0:
        if len(acquired_indices) > 0:
            acquired_inputs,acquired_outputs,acquired_modalities = self.acquire_unlabelled_samples(input_array,output_array,fraction,unlabelled_fraction,acquired_indices)
            inputs = np.concatenate((inputs,acquired_inputs),0)
            #print(acquired_labels)
            #""" Note - Acquired Labels from Network Predictions are Used, Not Ground Truth """
            #acquired_labels = np.fromiter(acquired_labels.values(),dtype=float)
            acquired_labels = np.array(list(acquired_labels.values()))
            acquired_labels = acquired_labels.reshape((-1,))
            ##""" For cold_gt trials, run this line """
            ##acquired_labels = np.array(list(acquired_labels.values()))
            ##acquired_labels = acquired_labels.reshape((-1,))
            ##print(outputs.shape,acquired_labels.shape)
            ##""" End GT Labels """
            
            #print(acquired_labels)
            outputs = np.concatenate((outputs,acquired_labels),0) 
        return inputs,outputs
    
    def list_of_color_transforms(self,s=1):
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        list_of_transforms = [rnd_color_jitter,rnd_gray]
        return list_of_transforms
    
    def list_of_crop_transforms(self):
        rnd_crop = transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0))
        return [rnd_crop]
    
    def __getitem__(self,index):
        true_index = self.remaining_indices[index] #this should represent indices in original unlabelled set
        input_frame = self.input_array[index]
        label = self.label_array[index]
        modality = self.modality_array[index]
        dataset = self.dataset_list[index] #task name

        if 'image' in self.modalities:
            input_frame = np.transpose(input_frame,(1,2,0)) #32 x 32 x 3, H x W x C
            input_frame = transforms.functional.to_pil_image(input_frame,mode='RGB') #output is 32 x 32 x 3
            #print(input_frame.size)
            list_of_transforms = []
            if self.input_perturbed == True:
                list_of_crop_transforms = self.list_of_crop_transforms()
                list_of_transforms.extend(list_of_crop_transforms)
                
                list_of_color_transforms = self.list_of_color_transforms(s=1)
                list_of_transforms.extend(list_of_color_transforms)
                
                transform = transforms.Compose(list_of_transforms)
            else:
                transform = transforms.Compose(list_of_transforms)
            
            list_of_transforms.extend([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            frame = transform(input_frame)

        else: #Time-Series Pathway
            if input_frame.dtype != float:
                input_frame = np.array(input_frame,dtype=float)
            
            if self.input_perturbed == True:
                
                if self.phase == 'test':
                    mult_factor = 1
                else:
                    mult_factor = 1
                
                """ Univariate Normal is Much Faster to Compute """
                if self.dataset_name == 'ptb':
                    variance_factor = 0.01*mult_factor
                elif self.dataset_name == 'uci_emg':
                    variance_factor = input_frame.max()/5 #new implementation April 10th, 2020 
                else:
                    variance_factor = 100*mult_factor
                gauss_noise = np.random.normal(0,variance_factor,size=(2500))
                input_frame = input_frame + gauss_noise
            
            """ Normalize Data Frame """
            if self.cl_scenario == 'Task-IL': #make sure all tasks are exposed to same input range
                #input_frame = (input_frame - np.min(input_frame))/(np.max(input_frame) - np.min(input_frame) + 1e-8)
                input_frame = input_frame
            else:
                if self.dataset_name not in ['cardiology','physionet2017','physionet2016','uci_emg']:# or self.dataset_name != 'physionet2017':# or self.dataset_name != 'cipa':
                    input_frame = (input_frame - np.min(input_frame))/(np.max(input_frame) - np.min(input_frame) + 1e-8)
            
            """ ESSENTIAL - Convert Data to Torch Tensor """
            frame = torch.tensor(input_frame,dtype=torch.float)
            label = torch.tensor(label,dtype=torch.float)
            
            """ Frame Input Has 1 Channel """
            frame = frame.unsqueeze(0)

        return frame,label,modality,dataset,true_index
        
    def __len__(self):
        #print(self.input_array[:5])
        #print(type(self.input_array))
        #print('Phase Array Shape: %i' % self.input_array.shape[0])
        return len(self.input_array)
