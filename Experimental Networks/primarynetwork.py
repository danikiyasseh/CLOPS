# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:20:50 2019

@author: DaniK
"""

import torch
import torch.nn as nn
#from hyperattention_module import HyperAttention
#from hyperattention_module_trunkmask import HyperAttention
from hyperattention_module_nonlocal import HyperAttention as HyperAttentionNL
from hyperattention_module_trunkmask import HyperAttention as HyperAttentionDual
from conv_module import conv_block

class PrimaryNetwork(nn.Module):
    
    def __init__(self,dropout_type,p1,p2,p3,dataset_name,stride=3,hyperattention_type='nonlocal',bptt_steps=0,heads='multi'):
        super(PrimaryNetwork,self).__init__()
        
        kernels = [7,7,7]
        channels = [1,4,16,32]
        dropouts = [p1,p2,p3]
        hidden_dims1 = list(map(lambda c:2*c,channels[1:])) #Hidden Vector Across Time
        hidden_dims2 = list(map(lambda c:2*c,channels[:-1])) #Hidden Vector Across Layers
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        """ Original Modules and Weights """
        original_modules = nn.ModuleList()
        """ Attention Modules to Generate Weights """
        attention_modules = nn.ModuleList()
        """ Attention Hidden States for Time """
        hidden_attentions_time = []
        """ Class For Forward Pass with New Params """
        conv_modules = nn.ModuleList()
        
        """ Choose HyperAttention Module Based on Specified Type """
        if hyperattention_type == 'nonlocal':
            hyperattention_module = HyperAttentionNL
        elif hyperattention_type == 'dual':
            hyperattention_module = HyperAttentionDual

        for l in range(len(kernels)):
            conv = nn.Conv1d(channels[l],channels[l+1],kernels[l],stride,bias=False)
            original_modules.append(conv)
            
            block = conv_block(channels[l+1],dropouts[l])
            conv_modules.append(block)

            attention_module = hyperattention_module(channels[l],channels[l+1],kernels[l],hidden_dims1[l],hidden_dims2[l])
            attention_modules.append(attention_module)
            
            hidden_attention_time = torch.rand((1,hidden_dims1[l]),device=device)
            hidden_attentions_time.append(hidden_attention_time)
            
        hidden_attention_layer = torch.rand((1,hidden_dims2[0]),device=device)
        
        head_input_dim = 100 #c4 for average poooling
        self.linear1 = nn.Linear(32*10,head_input_dim)
        
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
        elif heads == 'single':
            """ Single Head Continual Learning """
            self.single_head = nn.Linear(head_input_dim,5+4+12+1+1)

        self.relu = nn.ReLU()
        
        self.heads = heads
        self.dataset_name = dataset_name
        self.bptt_steps = bptt_steps
        self.original_modules = original_modules
        self.attention_modules = attention_modules
        self.hidden_attentions_time = hidden_attentions_time
        self.hidden_attention_layer = hidden_attention_layer
        self.conv_modules = conv_modules
    
    def apply_mask(self,param,param_attention):
        """ Take Diagonal Elements of Param Attention """
        #out_channel = param_attention.shape[0]
        #kernel_size = param_attention.shape[1]
        #channel_param_attention = torch.zeros((out_channel,kernel_size))
        #for out_channel,out_channel_attention in enumerate(param_attention):
        #    out_channel_attention = out_channel_attention.diag()
        #    channel_param_attention[out_channel] = out_channel_attention
            
        """ Normalize the Attention Maps """
        #channel_param_attention = self.softmax(channel_param_attention)
        
        """ Convert Attention to Mask """
        sorted_attention,sorted_indices = torch.sort(param_attention.flatten().abs())
        quantile_index = int(0.2*len(param_attention.flatten()))
        attention_cutoff = sorted_attention[quantile_index]
        channel_param_attention_mask = param_attention.abs() <= attention_cutoff
        #channel_param_attention_mask = torch.gt(param_attention,attention_cutoff)
        #channel_param_attention_mask = torch.where(channel_param_attention_mask==1,torch.tensor(0),torch.tensor(1))
        """ Apply Temporary Mask """
        for in_channel,in_channel_param in enumerate(param.permute((1,0,2))):
            """ Hard Mask """
            if channel_param_attention_mask.shape == (4,7,7):
                print(param_attention[:,0,:])
                print(param[:,in_channel,:])
            #wont change original param value for updating - good
            #param[:,in_channel,:].masked_fill_(channel_param_attention_mask[:,0,:],0)
            #param[:,in_channel,:].masked_fill_(param_attention[:,0,:].abs().gt(attention_cutoff),0)
            #wont change original param value for updating - good
            #param[:,in_channel,:].data.mul_(param_attention[:,0,:])
            #wont change original param value for updating - good
            #param[0,in_channel,:].data.copy_(F.linear(param[0,in_channel,:],param_attention[0,:,:]))
            #this gives you backprop error b/c of equality sign - bad
            #param[0,in_channel,:] = F.linear(param[0,in_channel,:],param_attention[0,:,:])
            if channel_param_attention_mask.shape == (4,7,7):
                print(param[:,in_channel,:])
            
            #""" Soft Mask """
            #param[:,in_channel,:] = in_channel_param * channel_param_attention
        
        return param
    
    def forward(self,x,bptt_counter):
        
        """ Obtain Updated Hidden Attentions Time For Next Pass """
        hidden_attentions_time_new = []
        """ Initial Hidden Attention (Layer) """
        hidden_attention_layer = self.hidden_attention_layer 
        """ Store Post Attention Param for Gradient Masking """
        params = []
        #""" Make Learning Decision Based on Param Attentions """
        #param_attentions = []
        
        """ i represents the layer number """
        for i,(original_module,attention_module,hidden_attention_time,conv_module) in enumerate(zip(self.original_modules,self.attention_modules,self.hidden_attentions_time,self.conv_modules)):
            """ j represents the number of distinct weights in layer - usually 1 """
            for j,param in enumerate(original_module.parameters()):
                #print(param.shape)
                #if i == 0:#and j == 0:
                #    print('Original')
                #    print(param[0])
                
                """ Time and Layer Hidden Attention """
                param, hidden_attention_time, hidden_attention_layer = attention_module(param,hidden_attention_time,hidden_attention_layer)
                params.append(param)
                #if i == 0:#and j == 0:
                    #print('Post Attention')
                    #print(param[0])
                    #print(hidden_attention_time.shape)
                    #print(hidden_attention_layer.shape)
                
                """ You Must Append Raw Tensor (i.e. .data) and Not Variable to Allow for Future Backprops (Otherwise Use retain_graph = True to BPTT for X timepoints) """
                if (bptt_counter+1) % (self.bptt_steps+1) == 0:
                    #retain_graph is False so needs raw tensor
                    hidden_attentions_time_new.append(hidden_attention_time.data)
                else:
                    #retain graph is true so no need for raw tensor
                    hidden_attentions_time_new.append(hidden_attention_time)
                #param_attentions.append(param_attention)
                #param = self.apply_mask(param,param_attention)
                #print(x.shape)
                #if i == 0:#and j == 0:
                #    print('Post Mask')
                #    print(param[0])
                x = conv_module(x,param)
        
        x = x.view((x.shape[0],-1))
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
        else:
            x = self.single_head(x)
        
        #doing this DOES update the hidden attentions for next pass
        self.hidden_attentions_time = hidden_attentions_time_new
        
        return x, params, hidden_attentions_time_new

    
        
        
        
        