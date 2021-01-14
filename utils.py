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
    Function to apply convolution on complex features

    Inputs:
        x_real - Real part of complex feature.
        x_img - Imaginary part of complex feature.
        conv_real - Convolution applied on the real part.
        conv_imag - Convolution applied on the imaginary part.
    Outputs:
        real_out - Convolved real part of complex feature.
        imag_out - Convolved imaginary part of complex feature.
    """
        
    real_out = conv_real(x_real) - conv_imag(x_imag)
    imag_out = conv_real(x_imag) + conv_imag(x_real)
    return real_out, imag_out

def complex_relu(x, device):
    """
    Function to apply ReLu on complex features

    Inputs:
        x - Batch of complex features. Shape: [B, ?, ?, ?]
        device - PyTorch device used to run the model on.
    Outputs:
        result - Resulting feature after ReLU. [B, ?, ?, ?]
    """
    
    #Zou dit werken? Sowieso
    c = torch.ones(x.shape, device=device)
    check = x[0,0,0,0]

    noemer = complex_norm(x)

    result  = noemer / torch.max(noemer, c)
    result = result * x
    return result


def complex_norm(x):
    """
    Function calculate norm of complex feature

    Inputs:
        x - Batch of complex features. Shape: [B, ?, ?, ?]
    Outputs:
        result - Norm of complex features. Shape: [B, ?, ?, ?]
    """
    
	result = torch.sqrt((x*x.conj()).real)
	return result 

def complex_max_pool(x, pool):
    """
    Function to apply MaxPool on complex features

    Inputs:
        x - Batch of complex features. Shape: [B, ?, ?, ?]
        ?
    Outputs:
        result - Resulting feature after MaxPool. [B, ?, ?, ?]
    """
    
	norm = complex_norm(x)
	iets, indices = pool(norm)

	zeros = torch.zeros_like(x)
	zeros[indices] = 1
	print(zeros.shape)
	#print(result.shape)
	result = torch.where(zeros >1, x, 0)
	print(result.shape)
	#result = x[indices]
	result = torch.index_select(x, 3,indices)
	#print(result[0,0,0,0])
	#print(result.shape)
	return result