###############################################################################
# MIT License
#
# Copyright (c) 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Authors: Luuk Kaandorp, Ward Pennink, Ramon Dijkstra, Reinier Bekkenutte 
# Date Created: 2020-01-08
###############################################################################

"""
Utility functions
"""

import torch
#import torch.linalg#
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def complex_conv(x_real, x_imag, conv_real, conv_imag):
    """
    Function to apply convolution on complex features.

    Inputs:
        x_real - Real part of complex feature.
        x_img - Imaginary part of complex feature.
        conv_real - Convolution to be applied on the real part.
        conv_imag - Convolution to be applied on the imaginary part.
    Outputs:
        real_out - Convolved real part of complex feature.
        imag_out - Convolved imaginary part of complex feature.
    """
    
    # calculate the convolution on the real and imaginary parts
    real_out = conv_real(x_real) - conv_imag(x_imag)
    imag_out = conv_real(x_imag) + conv_imag(x_real)
    
    # return the convolved real and imaginary parts
    return real_out, imag_out

def complex_relu(x, device, c=1):
    """
    Function to apply ReLu on complex features.

    Inputs:
        x - Batch of complex features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        device - PyTorch device used to run the model on.
        c - Fixed constant used in the max function. Default = 1
    Outputs:
        result - Resulting features after ReLU. [B, ?, ?, ?]
    """

    # calculate the denominator
    norm = complex_norm(x, device)

    #if model is RESNET: c = 1
    #else expectation of norm
    constant = torch.mean(norm, dim=-1, keepdim=True)


    #TODO MISSCHIEN SHAPES FIXEN IDKKKKKK

    # create the sum constant
    # constant = torch.ones(x.shape, device=device) * c

    # calculate the resulting features
    result = norm / torch.max(norm, constant)
    result = result * x
    
    # return the resulting features
    return result


def complex_norm(x, device):
    """
    Function calculate norm of complex features.

    Inputs:
        x - Batch of complex features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W - feature width
                H - feature height
    Outputs:
        norm - Norm of complex features. Shape: [B, C, W, H]
    """

    try:
        norm = torch.sqrt((x*x.conj()).real)
    except:
        # calculate the norm of a real valued feature
        norm = torch.abs(x)
    
    # return the resulting norm
    return norm

# TODO: below is not used anymore?
def complex_max_pool(x, device, pool):
    """
    Function to apply MaxPool on complex features.

    Inputs:
        x - Batch of complex features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        pool - 
    Outputs:
        result - Resulting feature after MaxPool. [B, ?, ?, ?]
    """
    norm = complex_norm(x, device)
    iets, indices = pool(norm)

    flattened_tensor = x.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

    #zeros = torch.zeros_like(x)
    #zeros[indices] = 1
    #print(zeros.shape)
    #print(result.shape)
    #result = torch.where(zeros >1, x, 0)
    #print(result.shape)
    #result = x.view(4,16,24*24)[indices]
    #result = torch.index_select(x, 3,indices)
    return output