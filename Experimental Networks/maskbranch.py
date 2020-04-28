# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:36:39 2019

@author: DaniK
"""

import torch.nn as nn

class MaskBranch(nn.Module):
    
    def __init__(self,in_channel,out_channel,kernel_size,hidden_dim1,hidden_dim2):
        super(MaskBranch,self).__init__()
        input_channels = in_channel*out_channel
        self.conv1 = nn.Conv1d(input_channels,2*input_channels,2,stride=2)
        self.batchnorm1 = nn.BatchNorm1d(2*input_channels)
        self.conv2 = nn.ConvTranspose1d(2*input_channels,input_channels,2,stride=2,output_padding=1)
        self.batchnorm2 = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        
        self.hiddentime2attention = nn.Linear(hidden_dim1,input_channels*kernel_size)
        self.hiddenlayer2attention = nn.Linear(hidden_dim2,input_channels*kernel_size)
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.hidden_dim1 = hidden_dim1
        
    def forward(self,param,h0=None,h1=None):
        if h0 is not None:
            prev_attention_time = self.hiddentime2attention(h0)
            prev_attention_time = prev_attention_time.view((1,self.input_channels,-1))

        if h1 is not None:
            prev_attention_layer = self.hiddenlayer2attention(h1)
            prev_attention_layer = prev_attention_layer.view((1,self.input_channels,-1))
        
        hmid = self.relu(self.batchnorm1(self.conv1(param)))
        hmid = self.batchnorm2(self.conv2(hmid))
        
        if h0 is not None and h1 is not None:
            hmid = self.relu(hmid + prev_attention_time + prev_attention_layer)
        elif h0 is None and h1 is not None:
            hmid = self.relu(hmid + prev_attention_layer)
        elif h1 is None and h0 is not None:
            hmid = self.relu(hmid + prev_attention_time)

        h2 = hmid.view((self.out_channel,self.in_channel,self.kernel_size))
        return h2
        
        