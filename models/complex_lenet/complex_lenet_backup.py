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
from utils import *

class LenetEncoder(nn.Module):
    """
	LeNet encoder model
	"""

    def __init__(self, k, device):
        """
        Encoder module of the network
        Inputs:
            k - Level of anonimity. k-1 fake features are generated
                to obscure the real feature.
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
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            training - Boolean value. Default = True
                True when training
                False when using in application
        Outputs:
            x - Real encoded feature. Shape: [B, C, W, H]
            thetas - Angles used for rotating the real feature. Shape: [B, 1, 1, 1]
            discriminator_predictions - Predictions from the discriminator. Shape: [B * k, 1]
            labels - Real labels of the encoded features. Shape: [B * k, 1]
        """
        
        # apply the first convolutional layer of the LeNet model 
        convolved_images = self.conv1(image_batch)

        # check if training
        if training:
            # encode the convolved images using the generator
            with torch.no_grad():
                a, x, thetas, fake_x, delta_thetas = self.generator(convolved_images)
            
            # create a batch of real feature and encoded fake features
            real_and_fake_images = torch.cat([a, fake_x], dim=0)
            
            # create the labels for the batch
            # 1 if real, 0 if fake
            labels = torch.cat([torch.ones(a.shape[0]),torch.zeros(fake_x.shape[0])], dim=0)
            labels = labels.reshape(labels.shape[0], 1)
            
            # predict the labels using the discriminator
            discriminator_predictions = self.discriminator(real_and_fake_images)

            # return the real encoded images, thetas, discriminator predictions and labels
            return x, thetas, discriminator_predictions, labels
        else:
            # encode the convolved images using the generator
            x, thetas = self.generator(convolved_images)
        
            # return the real encoded images and thetas
            return x, thetas

class EncoderGenerator(nn.Module):
    """
	Generator part of the LeNet encoder model
	"""
    
    def __init__(self, k, device):
        """
        Generator model of the encoder
        Inputs:
            k - Level of anonimity. k-1 fake features are generated
                to obscure the real feature.
            device - PyTorch device used to run the model on.
        """
        super().__init__()

        # save the inputs
        self.k = k
        self.device = device

    def forward(self, a, training=True):
        """
        Inputs:
            a - Input batch of convolved images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            training - Boolean value. Default = True
                True when training
                False when using in application
        Outputs:
            a - Real non-encoded feature. Shape: [B, C, W, H]
            x - Real encoded feature. Shape: [B, C, W, H]
            thetas - Angles used for rotating the real feature. Shape: [B, 1, 1, 1]
            fake_x - Fake generated features. Shape: [B * k-1, C, W, H]
            delta_thetas - Angles used for rotating the fake features. Shape: [B * k-1, 1, 1, 1]
        """

        # save the image dimensions for later use
        image_dimensions = a.shape
        
        # compute the magnitude of the image batch
        a_magnitude = torch.norm(a).item()

        # create real obfuscating features b
        b = torch.normal(0, 1, size=tuple((image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3])))
        b_magnitude = torch.sqrt(torch.sum(torch.square(b)).type(torch.FloatTensor))
        b = (b / b_magnitude)* a_magnitude
        b = b.to(self.device)

        # sample angles to rotate the features for the real rotation
        thetas = torch.Tensor(image_dimensions[0], 1, 1, 1).uniform_(0, 2 * np.pi).to(self.device)
        thetas = thetas.cpu()
        thetas = (1j * thetas).exp()
        thetas = thetas.to(self.device)
        
        # compute encoded real feature
        x = (a + b *1j) * thetas
        x = x.to(self.device)
        
        # check if training
        if training:      
            # create fake obfuscating features b
            fake_b = torch.normal(0, 1, size=tuple(((self.k-1) * image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3])))
            fake_b_magnitude = torch.sqrt(torch.sum(torch.square(fake_b)).type(torch.FloatTensor))
            fake_b = (fake_b / fake_b_magnitude)* a_magnitude
            fake_b = fake_b.to(self.device)
        
            # sample k-1 delta angles to rotate the features for fake examples
            delta_thetas = torch.Tensor((self.k-1) * image_dimensions[0], 1, 1, 1).uniform_(0, np.pi).to(self.device)
            delta_thetas = delta_thetas.cpu()
            delta_thetas = (1j * delta_thetas).exp()
            delta_thetas = delta_thetas.to(self.device)
        
            # compute encoded fake features
            fake_a = torch.cat([a]*(self.k-1),dim=0)
            fake_x = (a + fake_b *1j) * delta_thetas
            fake_x = fake_x.to(self.device)
            fake_x = fake_x.real
            
            # return real feature, real encoded feature, thetas, fake encoded feature and delta thetas
            return a, x, thetas, fake_x, delta_thetas
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
        """
        super().__init__()
        
        # initialize the linear layer
        self.linear = nn.Linear(6*28*28,1)
        
        # initialize the sigmoid layer
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoded_batch):
        """
        Inputs:
            encoded_batch - Input batch of encoded features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        Outputs:
			predictions - Predictions for real and fake feature. Shape: [B, 1]
                1 when real feature
                0 when fake feature
        """
        
        # reshape the batch
        encoded_batch = encoded_batch.view(encoded_batch.shape[0],6*28*28)
        
        # predict the labels
        predictions = self.linear(encoded_batch)
        predictions = self.sigmoid(predictions)
        
        # return the predictions
        return predictions
 
class LenetProcessingModule(nn.Module):
    """
	LeNet processing module model
	"""
    
    def __init__(self, device):
        """
        Processing module of the network
        Inputs:
            device - PyTorch device used to run the model on.
        """
        super(LenetProcessingModule, self).__init__()
        
        # save the inputs
        self.device = device
        
        # initialize the layers of the LeNet model
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True) 
        self.conv2_real = nn.Conv2d(6, 16, 5, bias=False)
        self.conv2_imag = nn.Conv2d(6, 16, 5, bias=False)                 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, encoded_batch):
        """
        Inputs:
            encoded_batch - Input batch of encoded features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        Outputs:
			processed_batch - Output batch of further processed features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        """
        
        # transform the encoded features using the model layers
        # TODO: make this work

        encoded_batch_real = encoded_batch.real
        encoded_batch_imag = encoded_batch.imag

        intermediate_real, intermediate_imag = complex_relu(encoded_batch_real,self.device), complex_relu(encoded_batch_imag,self.device)
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, self.pool), complex_max_pool(intermediate_imag, self.pool)

        intermediate_real, intermediate_imag = complex_conv(intermediate_real, intermediate_imag, self.conv2_real, self.conv2_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real,self.device), complex_relu(intermediate_imag,self.device)        
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, self.pool), complex_max_pool(intermediate_imag, self.pool)

        intermediate_real, intermediate_imag =  intermediate_real.view(-1, 16 * 5 * 5), intermediate_imag.view(-1, 16 * 5 * 5)

        intermediate_real, intermediate_imag = self.fc1(intermediate_real), self.fc1(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real,self.device), complex_relu(intermediate_imag,self.device)    

        intermediate_real, intermediate_imag = self.fc2(intermediate_real), self.fc2(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real,self.device), complex_relu(intermediate_imag,self.device)

        intermediate_real, intermediate_imag = self.fc3(intermediate_real), self.fc3(intermediate_imag)    
        # ### Forward with complex

        # intermediate_imag = complex_relu(encoded_batch_imag,self.device)
        # intermediate_imag = complex_max_pool(intermediate_imag,self.pool)

        # intermediate_real = 

        # imediate_imag = self.Conv2d
        # #print(x.shape)
        # indices = complex_max_pool(x,self.pool)
        # #print(x.shape)
        # x = x[indices]
        # # Split x in real and imaginary part
        # x_real, x_imag = self.complex_conv(x, conv2_imag, conv2_real)
        # x = complex_relu(x,self.device)
        # x = self.pool(x)

        # x = x.view(-1, 16 * 5 * 5)

        # x = self.fc1(x)
        # x = complex_relu(x,self.device)

        # x = self.fc2(x)
        # x = complex_relu(x,self.device)

        # x = self.fc3(x)
        x = intermediate_real+intermediate_imag
        #print("ZO KAN IK DAT WETEN", x.shape)
        return x

class LenetDecoder(nn.Module):
    """
	LeNet decoder model
	"""
    
    def __init__(self, num_classes):
        """
        Decoder module of the network
        """
        super(LenetDecoder, self).__init__()

        # initialize the softmax layer
        self.num_classes = num_classes
        #self.linear = nn.Linear(16*12*12, num_classes)

        self.softmax = nn.Softmax()


    def forward(self, encoded_batch, thetas):
        """
        Inputs:
            encoded_batch - Input batch of encoded features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
            thetas - Angles used for rotating the real feature. Shape: [B, 1, 1, 1]
        Outputs:
			decoded_batch - Output batch of decoded features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        """
        
    	# rotate the features back to their original state
        decoded_batch = encoded_batch * torch.exp(-1j * thetas.squeeze())[:, None]
        
        # get rid of the imaginary part of the complex features
        decoded_batch = decoded_batch.real
        
        # apply the softmax layer#
        decoded_batch = self.softmax(decoded_batch)
        
        # return the decoded batch
        return decoded_batch

class ComplexLenet(nn.Module):
    """
	Complex LeNet model
	"""
    
    def __init__(self, device, num_classes, k=2):
        """
        Complex LeNet network
        Inputs:
            device - PyTorch device used to run the model on.
            k - Level of anonimity. k-1 fake features are generated
                to obscure the real feature. Default = 2
        """
        super(ComplexLenet, self).__init__()

        # save the inputs
        self.num_classes = 10
        self.device = device
        self.k = k

        # initialize the different modules of the network
        self.encoder = LenetEncoder(self.k, self.device)
        self.proccessing_module = LenetProcessingModule(self.device)
        self.decoder = LenetDecoder(self.num_classes)

    def forward(self, image_batch, training=True):
        """
        Inputs:
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            training - Boolean value. Default = True
                True when training
                False when using in application
        Outputs:
			decoded_feature - Output batch of decoded real features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
            discriminator_predictions - Predictions from the discriminator. Shape: [B * k, 1]
            labels - Real labels of the encoded features. Shape: [B * k, 1]
        """
        
        # check if training
        if training:
            # run the image batch through the encoder (generator and discriminator)
            x, thetas, discriminator_predictions, labels = self.encoder(image_batch)
            
            # send the encoded feature to the processing unit
            x = self.proccessing_module(x)
            
            # decode the feature from
            x = self.decoder(x, thetas)
            
            # return the decoded feature, discriminator predictions and real labels
            return x, discriminator_predictions, labels
        else:
            # run the image batch through the encoder (only generator)
            x, thetas = self.encoder(image_batch)
            
            # send the encoded feature to the processing unit
            #x = self.proccessing_module(x)
            
            # decode the feature from
            #x = self.decoder(x, theta)
            
            # return the decoded feature 
            return x