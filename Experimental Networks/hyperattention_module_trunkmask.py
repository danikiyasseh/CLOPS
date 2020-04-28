# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:45:34 2019

@author: DaniK
"""

import torch.nn as nn
from trunkbranch import TrunkBranch
from maskbranch import MaskBranch

class HyperAttention(nn.Module):
	""" Takes in Parameters, Performs Trunk and Mask Attention and Returns Parameters
	
	Args:
		Model Parameters
	
	Returns: 
		Model Parameters
	"""
	
	def __init__(self,in_channel,out_channel,kernel_size,hidden_dim1,hidden_dim2):
		super(HyperAttention,self).__init__()
		self.trunkbranch = TrunkBranch(in_channel,out_channel,kernel_size)
		self.maskbranch = MaskBranch(in_channel,out_channel,kernel_size,hidden_dim1,hidden_dim2)
		
		self.attentiontime2hidden = nn.Linear(out_channel*in_channel*kernel_size,hidden_dim1)
		self.attentionlayer2hidden = nn.Linear(out_channel*in_channel*kernel_size,hidden_dim1)
		self.sigmoid = nn.Sigmoid()
		
		self.input_channels = in_channel*out_channel
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.hidden_dim1 = hidden_dim1
		self.hidden_dim2 = hidden_dim2
		
	def forward(self,param,h0=None,h1=None):
		param = param.view((1,self.input_channels,-1))
		trunk_output = self.trunkbranch(param)
		mask_output = self.maskbranch(param,h0,h1)
		param = (1 + mask_output) * trunk_output
		
		mask_output_reshaped = mask_output.view((1,-1))
		hidden_attention_time = self.sigmoid(self.attentiontime2hidden(mask_output_reshaped))
		hidden_attention_layer = self.sigmoid(self.attentionlayer2hidden(mask_output_reshaped))
		return param, hidden_attention_time, hidden_attention_layer
		
	
	
	