# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:45:34 2019

@author: DaniK
"""

import torch.nn as nn
from nonlocalblock import NonLocalBlock

class HyperAttention(nn.Module):
	""" Takes in Parameters, Performs Non-Local Attention and Returns Parameters
	
	Args:
		Model Parameters
	
	Returns: 
		Model Parameters
	"""
	
	def __init__(self,in_channel,out_channel,kernel_size,hidden_dim1,hidden_dim2):
		super(HyperAttention,self).__init__()
		self.nonlocalblock = NonLocalBlock(in_channel,out_channel,kernel_size,hidden_dim1,hidden_dim2)
		self.attentiontime2hidden = nn.Linear(kernel_size*kernel_size,hidden_dim1)
		self.attentionlayer2hidden = nn.Linear(kernel_size*kernel_size,hidden_dim1)
		self.sigmoid = nn.Sigmoid()
		
		self.input_channels = in_channel*out_channel
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.hidden_dim1 = hidden_dim1
		
	def forward(self,param,h0=None,h1=None):
		param = param.view((1,self.input_channels,-1))
		param, attention = self.nonlocalblock(param,h0,h1)
		attention_reshaped = attention.view((1,-1))
		hidden_attention_time = self.sigmoid(self.attentiontime2hidden(attention_reshaped))
		hidden_attention_layer = self.sigmoid(self.attentionlayer2hidden(attention_reshaped))
		return param, hidden_attention_time, hidden_attention_layer
		
	
	
	
	