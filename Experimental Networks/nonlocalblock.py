# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 21:45:11 2019

@author: DaniK
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    
    def __init__(self,in_channel,out_channel,kernel_size,hidden_dim1,hidden_dim2):
        super(NonLocalBlock,self).__init__()
        
        input_channels = in_channel*out_channel
        self.queryconv = nn.Conv1d(input_channels,input_channels//2,1)
        self.querybatchnorm = nn.BatchNorm1d(input_channels//2)
        
        self.keyconv = nn.Conv1d(input_channels,input_channels//2,1)
        self.keybatchnorm = nn.BatchNorm1d(input_channels//2)
        
        self.valueconv = nn.Conv1d(input_channels,input_channels//2,1)
        self.valuebatchnorm = nn.BatchNorm1d(input_channels//2)
        
        self.finalconv = nn.Conv1d(input_channels//2,input_channels,1)
        self.relu = nn.ReLU()
        
        self.hiddentime2attention = nn.Linear(hidden_dim1,kernel_size)
        self.hiddenlayer2attention = nn.Linear(hidden_dim2,kernel_size)
        
        self.input_channels = input_channels
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        
    def forward(self,param,h0=None,h1=None):
        query = self.querybatchnorm(self.queryconv(param))
        query = query.view((-1,self.kernel_size))
        
        key = self.keybatchnorm(self.keyconv(param))
        key = key.view((-1,self.kernel_size))
        
        querykey = torch.matmul(query.t(),key)

        if h0 is not None:
            hidden_attention_time = self.hiddentime2attention(h0)

        if h1 is not None:
            hidden_attention_layer = self.hiddenlayer2attention(h1)
        
        if h0 is not None and h1 is not None:
            querykey = querykey*hidden_attention_time*hidden_attention_layer
        elif h0 is None and h1 is not None:
            querykey = querykey*hidden_attention_layer
        elif h1 is None and h0 is not None:
            querykey = querykey*hidden_attention_time
            
        querykey_map = F.softmax(querykey,dim=-1)
        
        value = self.valuebatchnorm(self.valueconv(param))
        value = value.view((self.kernel_size,-1))
        output = torch.matmul(querykey_map,value)
        output = output.view((1,self.input_channels//2,self.kernel_size))
        output = self.finalconv(output)
        """ Skip Connection """
        param = param + output 
        param = param.view((self.out_channel,self.in_channel,self.kernel_size))
        return param, querykey_map
        
        
        
        
        
        
        