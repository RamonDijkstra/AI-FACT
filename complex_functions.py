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

# basic imports
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_conv(x_real, x_imag, conv_real, conv_imag):
    """
    Function to apply convolution on complex features.

    Inputs:
        x_real - Real part of complex feature.
        x_imag - Imaginary part of complex feature.
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

def complex_relu(x_real, x_imag, device, c=None):
    """
    Function to apply ReLu on complex features.

    Inputs:
        x_real - Real part of complex feature.
        x_imag - Imaginary part of complex feature.
        device - PyTorch device used to run the model on.
        c - Fixed constant used in the max function. Default = None
    Outputs:
        real_out - Real part of complex feature after ReLu.
        imag_out - Imaginary part of complex feature after ReLu.
    """

    # calculate the norm
    norm = complex_norm(x_real,x_imag)

    # check whether to give the given value of c
    if c is not None:
        # use the value of c for the constant
        constant = torch.ones(x_real.shape, device=device) * c
    else:
        # use the expectation of the norm
        constant = torch.mean(norm, dim=-1, keepdim=True)

    # calculate the resulting features
    result = norm / torch.max(norm, constant)
    real_out = result * x_real
    imag_out = result * x_imag

    # return the resulting features
    return real_out, imag_out

def complex_norm(x_real, x_imag):
    """
    Function calculate norm of complex and real features.

    Inputs:
        x_real - Real part of complex feature.
        x_imag - Imaginary part of complex feature.
    Outputs:
        norm - Norm of complex features.
    """

    # calculate the norm of the complex feature
    norm = torch.sqrt(x_real**2 + x_imag**2)

    # return the resulting norm
    return norm

def complex_max_pool(x_real, x_imag, pool):
    """
    Function to apply MaxPool on complex features.

    Inputs:
        x_real - Real part of complex feature.
        x_imag - Imaginary part of complex feature.
        pool - Standard PyTorch MaxPool module.
    Outputs:
        real_out - Real part of complex feature after MaxPool.
        imag_out - Imaginary part of complex feature after MaxPool.
    """

    # calculate the norm
    norm = complex_norm(x_real, x_imag)

    # retrieve the indices of the maxpool
    _, indices = pool(norm)

    # retrieve the associated values of the indices with the highest norm
    flattened_tensor_real = x_real.flatten(start_dim=2)
    flattened_tensor_imag = x_imag.flatten(start_dim=2)
    real_out = flattened_tensor_real.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    imag_out = flattened_tensor_imag.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

    # return the values with the highest norm
    return real_out, imag_out

def complex_batchnorm(x_real, x_imag):
    """
    Function to apply batch normalization on complex features.

    Inputs:
        x_real - Real part of complex feature.
        x_imag - Imaginary part of complex feature.
    Outputs:
        real_out - Real part of complex feature after BatchNorm.
        imag_out - Imaginary part of complex feature after BatchNorm.
    """

    # calculate the factor by which we need to divide
    denominator_real = x_real.view(x_real.shape[0], -1)
    denominator_imag = x_imag.view(x_imag.shape[0], -1)
    denominator_real = complex_norm(denominator_real, denominator_imag)**2
    denominator_imag = complex_norm(denominator_real, denominator_imag)**2
    denominator_real = denominator_real.mean(dim=1)
    denominator_imag = denominator_imag.mean(dim=1)
    denominator_real = torch.sqrt(denominator_real)
    denominator_imag = torch.sqrt(denominator_imag)
    denominator_real = denominator_real.view(denominator_real.shape[0],1,1,1)
    denominator_imag = denominator_imag.view(denominator_imag.shape[0],1,1,1)

    # calculate the normalized batches
    real_out = x_real/denominator_real
    imag_out = x_imag/denominator_imag

    # return the normalized batches
    return real_out, imag_out
