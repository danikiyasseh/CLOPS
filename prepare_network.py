#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:10:34 2020

@author: Dani Kiyasseh
"""
import torch
import torch.nn as nn

#%%
c1 = 1 #b/c single time-series
c2 = 4 #4
c3 = 16 #16
c4 = 32 #32
k=7 #kernel size #7 
s=3 #stride #3
#num_classes = 3

class cnn_network_time(nn.Module):
    
    """ CNN Implemented in Original Paper - Supposedly Simple but Powerful """
    
    def __init__(self,dropout_type,p1,p2,p3,dataset_name,hyperattention_type=None,bptt_steps=None,heads='multi',setting='Domain-IL',trial=''):
        super(cnn_network_time,self).__init__()
        
        self.conv1 = nn.Conv1d(c1,c2,k,s)
        self.batchnorm1 = nn.BatchNorm1d(c2)
        self.conv2 = nn.Conv1d(c2,c3,k,s)
        self.batchnorm2 = nn.BatchNorm1d(c3)
        self.conv3 = nn.Conv1d(c3,c4,k,s)
        self.batchnorm3 = nn.BatchNorm1d(c4)
        self.linear1 = nn.Linear(c4*10,100)
        
        head_input_dim = 100 #c4 for average pooling
        
        if heads == 'multi':
            """ Multi-Head Continual Learning """
            self.physio_head = nn.Linear(head_input_dim,5) #physionethead 
            self.bidmc_head = nn.Linear(head_input_dim,1)
            self.mimic_head = nn.Linear(head_input_dim,1)
            self.cipa_head = nn.Linear(head_input_dim,7)
            self.cardiology_head = nn.Linear(head_input_dim,12)
            self.physio2017_head = nn.Linear(head_input_dim,4)
            self.tetanus_head = nn.Linear(head_input_dim,1)
            self.ptb_head = nn.Linear(head_input_dim,1)
            self.fetal_head = nn.Linear(head_input_dim,1)
            #self.physio2017_head = nn.Linear(head_input_dim,1)
            self.physio2020_head = nn.Linear(head_input_dim,9)
            self.chapman_head = nn.Linear(head_input_dim,4)
        elif heads == 'single':
            """ Single Head Continual Learning """
            if setting == 'Task-IL':
                self.single_head = nn.Linear(head_input_dim,12+4)#12) #single head for Task-IL
            elif setting == 'Domain-IL':
                self.single_head = nn.Linear(head_input_dim,9)#*12) #9 classes for physio2020 for 12 leads #alternate = only 9 classes 
            elif setting == 'Class-IL':
                if trial == 'multi_task_learning':
                    self.single_head = nn.Linear(head_input_dim,12)
                else:
                    self.single_head = nn.Linear(head_input_dim,1)
            elif setting == 'Time-IL':
                self.single_head = nn.Linear(head_input_dim,4)
    
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool1d(2)
        
        if dropout_type == 'drop1d':
            self.dropout1 = nn.Dropout(p=p1) #0.2 drops pixels following a Bernoulli
            self.dropout2 = nn.Dropout(p=p2) #0.2
            self.dropout3 = nn.Dropout(p=p3)
        elif dropout_type == 'drop2d':
            self.dropout1 = nn.Dropout2d(p=p1) #drops channels following a Bernoulli
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)
        
        self.dataset_name = dataset_name
        self.heads = heads
                
    def forward(self,x):
        x = self.dropout1(self.maxpool(self.relu(self.batchnorm1(self.conv1(x)))))
        x = self.dropout2(self.maxpool(self.relu(self.batchnorm2(self.conv2(x)))))
        x = self.dropout3(self.maxpool(self.relu(self.batchnorm3(self.conv3(x)))))
        #x = torch.mean(x,dim=2) #average pooling
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        x = self.relu(self.linear1(x))
        
        if self.heads == 'multi':
            if self.dataset_name == 'physionet':
                x = self.physio_head(x)
            elif self.dataset_name == 'bidmc':
                x = self.bidmc_head(x)
            elif self.dataset_name == 'mimic':
                x = self.mimic_head(x)
            elif self.dataset_name == 'cipa':
                x = self.cipa_head(x)
            elif self.dataset_name == 'cardiology':
                x = self.cardiology_head(x)
            elif self.dataset_name == 'physionet2017':
                x = self.physio2017_head(x)
            elif self.dataset_name == 'tetanus':
                x = self.tetanus_head(x)
            elif self.dataset_name == 'ptb':
                x = self.ptb_head(x)
            elif self.dataset_name == 'fetal':
                x = self.fetal_head(x)
            elif self.dataset_name == 'physionet2016':
                x = self.physio2017_head(x)
            elif self.dataset_name == 'physionet2020':
                x = self.physio2020_head(x)
            elif self.dataset_name == 'chapman':
                x = self.chapman_head(x)
        elif self.heads == 'single':
            x = self.single_head(x)
        
        return x         
