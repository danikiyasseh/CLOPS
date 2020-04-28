#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 09:53:16 2019

@author: Dani Kiyasseh

Dataloader that Loads Images Only 
"""

from torch.utils.data import Dataset
from PIL import Image
import os


class my_dataset(Dataset):
    
    def __init__(self,basepath,task,task_data,phase,representation,order,colourmap,dim,transforms,modality):
        """ contains indices in original array for a particular phase """
        self.indices = task_data['indices'][phase]
        """ original full array that contains all data """
        self.inputs = task_data['inputs']
        self.outputs = task_data['outputs']
        """ define representation e.g. spectrogram """
        self.representation = representation
        """ define order of fusion """
        self.order = order
        """ define H and W dimension of image """
        self.dim = dim
        """ define colourmap of image """
        self.colourmap = colourmap
        """ define input transforms i.e. data augmentation """
        self.transforms = transforms
        """ define task at hand e.g. hfm ppg """
        self.task = task
        """ define modality e.g. ppg """
        self.modality = modality
        self.basepath = basepath
    
    def __getitem__(self,index):
        """ obtain correct index from list of indices """
        chosen_index = self.indices[index]
        label = self.outputs[chosen_index]
        """ Images were Saved as 'chosen_index_representation.png' e.g. '5_scalogram.png' """
        image_folder_path = os.path.join(self.basepath,self.modality,self.representation,self.colourmap,self.task)
        image_name = str(chosen_index) + '_' + self.representation + '.png'
        image_path = os.path.join(image_folder_path,image_name)
        """ Load Image """
        image = Image.open(image_path)            
        input_tensor = self.transforms(image)
        
        return input_tensor,label,chosen_index
        
    def __len__(self):
        return len(self.indices)

















