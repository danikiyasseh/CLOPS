# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:12:22 2019

@author: DaniK
"""

import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    
    def __init__(self,out_channel,dropout=0.1,stride=3):
        super(conv_block,self).__init__()
        self.bn = nn.BatchNorm1d(out_channel)
        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.stride = stride
        
    def forward(self,x,filters):
        x = F.relu(self.bn(F.conv1d(x,filters,stride=self.stride)))
        x = self.dropout(self.maxpool(x))
        return x