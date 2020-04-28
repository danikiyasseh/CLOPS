# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:08:56 2019

@author: DaniK
"""
import torch

def obtain_masks_for_original(params,percentile=0.2):
    """ Obtain Mask Based on Magnitude of Params """
    masks = []
    for param in params:
        nparams = param.nelement()
        index = int(nparams*percentile)-1
        val,ind = torch.sort(param.abs().flatten())
        cutoff_value = val[index]
        mask = param.abs() < cutoff_value
        masks.append(mask)
    return masks

def mask_gradients(net,masks):
    """ Mask Gradients Based on Supplied Masks """
    nmasked = 0
    for name,param in net.named_parameters():
        if 'original_modules' in name:
            #print(param.grad.sum())
            param.grad.masked_fill_(masks[nmasked],0)
            nmasked += 1
            #print(param.grad.sum())

#%%

def obtain_masks_for_all(params_dict,percentile=0.2):
    """ Obtain Mask Based on Magnitude of All Model Params (Original) """
    masks = dict()
    #for param in net.parameters():
    for name,param in params_dict.items():
        #if param.grad is not None:
        nparams = param.nelement()
        if nparams > 1:
            index = int(nparams*percentile)-1
            val,ind = torch.sort(param.abs().flatten())
            cutoff_value = val[-index]
            mask = param.abs() > cutoff_value
            #print(mask)
            masks[name] = mask
    return masks        

def mask_gradients_all(params_dict,masks_dict):
    """ Mask Gradients Based on Supplied Masks """
    #nmasked = 0
    #for name,param in net.named_parameters():
    for name1,param in params_dict.items():
        #if param.grad is not None:
        nparams = param.nelement()
        if nparams > 1:
            #print(param.grad.sum())
            #param.grad.masked_fill_(masks[nmasked],0)
            #print('grad')
            if name1 in masks_dict.keys():
                mask = masks_dict[name1]
                #print(param.grad)
                #print(param.grad.shape)
                #print(mask.shape)
                #print('-')
                param.grad.masked_fill_(mask,0)
                #print(param.grad)
                    #nmasked += 1
                #print(param.grad.sum())
