# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:45:34 2019

@author: DaniK
"""

import torch.nn as nn

class HyperAttention(nn.Module):
	""" Takes in Parameters, Performs Multi-Head Attention and Returns Parameters
	
	Args:
		Model Parameters
	
	Returns: 
		Model Parameters
		Attention Matrix 
	"""
	
	def __init__(self,in_channel,out_channel,kernel_size,hidden_dim):
		super(HyperAttention,self).__init__()
		embed_dim = in_channel
		if in_channel == 1:
			num_heads = 1
		else:
			num_heads = embed_dim//2
		print(embed_dim)
		self.attention = nn.MultiheadAttention(embed_dim,num_heads)
		self.attention2hidden = nn.Linear(out_channel*kernel_size*kernel_size,hidden_dim)
		self.hidden2attention = nn.Linear(hidden_dim,out_channel*kernel_size*kernel_size)
		self.hidden2hidden = nn.Linear(hidden_dim,hidden_dim)
		self.layernorm = nn.LayerNorm([out_channel,in_channel])
		self.sigmoid = nn.Sigmoid()
		
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.hidden_dim = hidden_dim
		
	def forward(self,param,h0):
		#param.reshape(self.kernel_size,self.out_channel,self.in_channel)
		""" Reshape Param for Multihead Attention """
		param = param.permute((2,0,1))
		#print(param.shape)
		param_output,param_attention = self.attention(param,param,param)
		param = self.layernorm(param_output + param)
		#print(param_attention.shape)
		""" Obtain Modified Param Attention and Hidden Attention """
		param_attention_reshaped = param_attention.reshape((1,self.out_channel*self.kernel_size*self.kernel_size))
		h1 = self.attention2hidden(param_attention_reshaped)
		h2 = self.sigmoid(h0 + h1)
		param_attention = self.sigmoid(self.hidden2attention(h2))
		param_attention = param_attention.reshape((self.out_channel,self.kernel_size,self.kernel_size))
		hidden_attention = self.hidden2hidden(h2)
		
		""" Reshape Param Back to Original """
		param = param.permute((1,2,0))
		#print(param.shape)
		
		#for in_channel,in_channel_param in enumerate(param.permute((1,0,2))):
		#	param[:,in_channel,:].data.mul_(param_attention[:,0,:])

		return param, param_attention, hidden_attention
		