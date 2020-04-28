# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:12:53 2019

@author: DaniK
"""

import torch.nn as nn

class TrunkBranch(nn.Module):
	
	def __init__(self,in_channel,out_channel,kernel_size):
		super(TrunkBranch,self).__init__()
		input_channels = in_channel*out_channel
		self.conv1 = nn.Conv1d(input_channels,input_channels,1)
		self.batchnorm1 = nn.BatchNorm1d(input_channels) #bn with one sample, problematic?
		self.conv2 = nn.Conv1d(input_channels,input_channels,1)
		self.batchnorm2 = nn.BatchNorm1d(input_channels)
		self.relu = nn.ReLU()
		
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.input_channels = input_channels
		
	def forward(self,param):
		#param = param.view((1,self.input_channels,-1))
		h1 = self.relu(self.batchnorm1(self.conv1(param)))
		h2 = self.relu(self.batchnorm2(self.conv2(h1)))
		h2 = h2.view((self.out_channel,self.in_channel,self.kernel_size))
		return h2
		
		
		
		
		
		