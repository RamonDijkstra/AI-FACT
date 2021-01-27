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
        x_img - Imaginary part of complex feature.
        conv_real - Convolution to be applied on the real part.
        conv_imag - Convolution to be applied on the imaginary part.
    Outputs:
        real_out - Convolved real part of complex feature.
        imag_out - Convolved imaginary part of complex feature.
    """
    
    # calculate the convolution on the real and imaginary parts
    # real_out = conv_real(x_real) - conv_imag(x_imag)
    # imag_out = conv_real(x_imag) + conv_imag(x_real)
    real_out = conv_real(x_real)
    imag_out = conv_imag(x_imag)

    # return the convolved real and imaginary parts
    return real_out, imag_out

# def complex_relu(x, device, x_complex=None, c=None):
#     """
#     Function to apply ReLu on complex features.

#     Inputs:
#         x - Batch of complex features. Shape: [B, C, W, H]
#                 B - batch size
#                 C - channels per feature
#                 W- feature width
#                 H - feature height
#         device - PyTorch device used to run the model on.
#         c - Fixed constant used in the max function. Default = None
#     Outputs:
#         result - Resulting features after ReLU. [B, C, W, H]
#     """

#     # TODO is this correct..?
#     #norm_x = x_complex if x_complex is not None else x
    
#     # calculate the denominator
#     norm = complex_norm(norm_x)

#     # check whether to give the given value of c
#     if c is not None:
#         # use the value of c for the constant
#         constant = torch.ones(x.shape, device=device) * c
#     else:
#         # use the expectation of the norm
#         constant = torch.mean(norm, dim=-1, keepdim=True)

#     # calculate the resulting features
#     result = norm / torch.max(norm, constant)
#     result = result * x
    
#     # return the resulting features
#     return result

    ##########CHAOS

def complex_relu(x_real, x_imag, device, c=None):
    """
    Function to apply ReLu on complex features.

    Inputs:
        x - Batch of complex features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        device - PyTorch device used to run the model on.
        c - Fixed constant used in the max function. Default = None
    Outputs:
        result - Resulting features after ReLU. [B, C, W, H]
    """
    
    # calculate the denominator
    #x = torch.complex(x_real,x_imag)
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
    result_real = result * x_real 
    result_imag = result * x_imag

    
    # return the resulting features
    return result_real, result_imag


    ############ /CHAOS



def complex_norm(x_real, x_imag):
    """
    Function calculate norm of complex and real features.

    Inputs:
        x - Batch of complex features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W - feature width
                H - feature height
    Outputs:
        norm - Norm of complex features. Shape: [B, C, W, H]
    """

  #   try:
		# # calculate the norm of complex valued features
  #       norm = torch.sqrt((x*x.conj()).real)
  #   except:
  #       # calculate the norm of a real valued feature
  #       norm = torch.abs(x)

    norm = torch.sqrt(x_real**2 + x_imag**2)
    
    # return the resulting norm
    return norm


    ########### CHAOS
def complex_max_pool(x_real, x_imag, pool):
    """
    Function to apply MaxPool on complex features.

    Inputs:
        x - Batch of complex features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        pool - Standard PyTorch MaxPool module.
    Outputs:
        result - Resulting feature after MaxPool. Shape: [B, C, W, H]
    """
    
    # calculate the norm
    norm = complex_norm(x_real,x_imag)    
    # retrieve the indices of the maxpool
    _, indices = pool(norm)

    # retrieve the associated values of the indices with the highest norm
    flattened_tensor_real = x_real.flatten(start_dim=2)
    flattened_tensor_imag = x_imag.flatten(start_dim=2)
    
    output_real = flattened_tensor_real.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    output_imag = flattened_tensor_imag.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

    # return the values with the highest norm
    return output_real, output_imag

#################### /CHAOS

# def complex_max_pool(x, pool):
#     """
#     Function to apply MaxPool on complex features.

#     Inputs:
#         x - Batch of complex features. Shape: [B, C, W, H]
#                 B - batch size
#                 C - channels per feature
#                 W- feature width
#                 H - feature height
#         pool - Standard PyTorch MaxPool module.
#     Outputs:
#         result - Resulting feature after MaxPool. Shape: [B, C, W, H]
#     """
	
# 	# calculate the norm
#     norm = complex_norm(x)
	
# 	# retrieve the indices of the maxpool
#     _, indices = pool(norm)

# 	# retrieve the associated values of the indices with the highest norm
#     flattened_tensor = x.flatten(start_dim=2)
#     output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

#     # return the values with the highest norm
#     return output

def complex_batchnorm(x_real, x_imag):
    """
    Function to apply batch normalization on complex features.

    Inputs:
        x - Batch of complex features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
    Outputs:
        result - Resulting feature after batch normalization. [B, C, W, H]
    """
    
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

    result_real = x_real/denominator_real
    result_imag = x_imag/denominator_imag

    return result_real, result_imag
