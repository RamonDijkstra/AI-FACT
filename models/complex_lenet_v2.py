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
Complex LeNet model
"""

# Adapted from https://github.com/wavefrontshaping/complexPyTorch
# from complexPyTorch-master.complexFunctions import *
# from complexPyTorch-master.complexLayers import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LenetEncoder(nn.Module):
    """
	LeNet encoder model
	"""

    def __init__(self, k, device):
        """
        Encoder model of the network

        Inputs:
            k - Number of fake features to generate in order to
                obscure the real features.
            device - PyTorch device used to run the model on.
        """
        super(LenetEncoder, self).__init__()
        
        # save the inputs
        self.k = k
        self.device = device

        # initialize the first 2d convolutional layer of the LeNet model
        self.conv1 = nn.Conv2d(3, 6, 5)

        # initialize the generator and discriminator for the encoder
        self.generator = EncoderGenerator(k, device)
        self.discriminator = EncoderDiscriminator(device)

    def forward(self, image_batch, training=True):
        """
        Inputs:
            image_batch - Input batch of images. Shape: [B, ?, ?, ?]
            training - Boolean value. 
                True when training
                False when using in application
        Outputs:
			reconstructed_image - Generated original image of shape 
				[B,image_shape[0],image_shape[1],image_shape[2]]
        """
        
        # apply the first convolutional layer of the LeNet model 
        convolved_images = self.conv1(image_batch)
        
        # encode the convolved images using the generator
        encoded_images, thetas = self.generator(convolved_images)
        
        # generate the labels for the encoded images
        # TODO: make this depend on K and shuffle the features
        labels = torch.cat([torch.ones(encoded_images.shape[0]),torch.zeros(encoded_images.shape[0])], dim=0)

        # check if training
        if training:
            # predict the labels using the discriminator
            discriminator_predictions = self.discriminator(encoded_images)
            
            # return the real encoded images, thetas, discriminator predictions and labels
            return generated[:a.shape[0],:,:,:], thetas, out, labels
        else:
            # return the real encoded images, thetas and labels
            return generated[:a.shape[0],:,:,:], thetas, labels

class EncoderGenerator(nn.Module):
    """
	Generator part of the LeNet encoder model
	"""
    
    def __init__(self, k, device):
        """
        Generator model of the encoder

        Inputs:
            k - Number of fake features to generate in order to
                obscure the real features.
            device - PyTorch device used to run the model on.
        """
        super().__init__()

        # save the inputs
        self.k = k
        self.device = device

    def forward(self, image_batch, training=True):
        """
        Inputs:
            image_batch - Input batch of images. Shape: [B, ?, ?, ?]
        Outputs:
			reconstructed_image - Generated original image of shape 
				[B,image_shape[0],image_shape[1],image_shape[2]]
            training - Boolean value. 
                True when training
                False when using in application
        """

        # save the image dimensions for later use
        image_dimensions = image_batch.shape

        # compute the magnitude of the image batch
        batch_magnitude = torch.norm(image_batch).item()

        # create real obfuscating features b
        b = torch.normal(0, 1, size=tuple((image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3])))
        b_magnitude = torch.sqrt(torch.sum(torch.square(b)).type(torch.FloatTensor))
        b = (b / b_magnitude)* batch_magnitude
        b = b.to(self.device)

        # sample angles to rotate the features for the real rotation
        thetas = torch.Tensor(image_dimensions[0], 1, 1, 1).uniform_(0, 2 * np.pi).to(self.device)
        thetas = thetas.cpu()
        thetas = (1j * thetas).exp()
        thetas = thetas.to(self.device)
        # thetas = thetas.reshape(self.k*a_size[0],1,1,1)
        
        # compute encoded real feature
        x = (a + b *1j) * thetas
        x = x.to(self.device)
        
        # check if training
        if training:      
            # create fake obfuscating features b
            fake_b = torch.normal(0, 1, size=tuple(((self.k-1) * image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3])))
            fake_b_magnitude = torch.sqrt(torch.sum(torch.square(fake_b)).type(torch.FloatTensor))
            fake_b = (fake_b / fake_b_magnitude)* batch_magnitude
            fake_b = fake_b.to(self.device)
        
            # sample k-1 delta angles to rotate the features for fake examples
            delta_thetas = torch.Tensor((self.k-1) * image_dimensions[0], 1, 1, 1).uniform_(0, np.pi).to(self.device)
            delta_thetas = delta_thetas.cpu()
            delta_thetas = (1j * delta_thetas).exp()
            delta_thetas = delta_thetas.to(self.device)
        
            # compute encoded fake features
            fake_a = torch.cat([a]*(self.k-1),dim=0)
            fake_x = (a + fake_b *1j) * fake_thetas
            fake_x = fake_x.to(self.device)
            
            # return the real encoded features, thetas, fake encoded features and delta thetas
            return x, thetas, fake_x, delta_thetas
        else:
            # return the real encoded features and thetas
            return x, thetas

class EncoderDiscriminator(nn.Module):
    """
	Discriminator part of the LeNet encoder model
	"""    
    
	def __init__(self, device):
		"""
        Discriminator model of the encoder

        Inputs:
            device - PyTorch device used to run the model on.
        """
		super().__init__()
        
        # save the inputs
        self.k = k
        
        # initialize the linear layer
		self.linear = nn.Linear(6*28*28,1)
        
        # initialize the sigmoid layer
		self.sigmoid = nn.Sigmoid()

	def forward(self,generated, a):
        """
        Inputs:
            image_batch - Input batch of images. Shape: [B, ?, ?, ?]
            training - Boolean value. 
                True when training
                False when using in application
        Outputs:
			reconstructed_image - Generated original image of shape 
				[B,image_shape[0],image_shape[1],image_shape[2]]
        """
        
		a = a.view(a.shape[0],6*28*28)
		generated= generated[a.shape[0]:,:,:]
		generated=generated.real
		generated = generated.view(generated.shape[0],6*28*28)
		x = torch.cat((a,generated),dim=0)
		x = self.linear(x)
		return self.sigmoid(x).squeeze()