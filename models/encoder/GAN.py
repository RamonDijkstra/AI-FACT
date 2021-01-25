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
GAN encoder model
"""

# basic imports
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class EncoderGAN(nn.Module):
    """
	Encoder GAN model.
	"""

    def __init__(self, encoding_layer, discrim_shape, k=2, lr=3e-4):
        """
        GAN model used as the encoder of the complex networks.

        Inputs:
            encoding_layer - PyTorch module (e.g. sequential) representing 
                the first layers of the model in the encoder.
            discrim_shape - Size of the discriminator input.
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. Default = 2
            lr - Learning rate to use for the optimizer. Default = 3e-4
        """
        super().__init__()
        
        # save inputs
        self.k = k
        self.lr = lr

        # initialize the generator and discriminator
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        self.generator = Generator(device, encoding_layer, k)
        self.discriminator = Discriminator(device, discrim_shape)

    def configure_optimizers(self):
        """
        Function to configure the optimizers.
        """
        
        # initialize optimizer for both generator and discriminator
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        # return the optimizers
        return [optimizer_gen, optimizer_disc], []

    def forward(self, image_batch, optimizer_idx):
        """
        GAN forward function.
        
        Inputs:
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            optimizer_idx - Int indicating the index of the current optimizer
                0 - GAN generator optimizer
                1 - GAN discriminator optimizer
                2 - Full model optimizer
        """
        
        # include the discriminator if the optimizer_idx=1 (discriminator optimizer)
        if optimizer_idx == 1:
            loss, x, thetas = self.discriminator_step(image_batch)
        else:
            loss, x, thetas = self.generator_step(image_batch)
        
        # return the loss, real encoded feature and real angles
        return loss, x, thetas

    def generator_step(self, image_batch):
        """
        Generator forward step.
        
        Inputs:
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
        """

        # encode the convolved images using the generator
        with torch.no_grad():
            a, x, thetas, fake_x, delta_thetas = self.generator(image_batch)
        
        # create labels for the fake features
        labels_fake = torch.ones([fake_x.shape[0], 1],requires_grad=True).to(self.device)
        
        # let the discriminator predict the fake fatures
        preds = self.discriminator(fake_x)
        
        # compute the loss over the fake features
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds, labels_fake)

        # return the generator loss, real encoded feature and real angles
        return loss, x, thetas

    def discriminator_step(self, image_batch):
        """
        Discriminator forward step.
        
        Inputs:
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
        """

        # encode the convolved images using the generator
        with torch.no_grad():
            a, x, thetas, fake_x, delta_thetas = self.generator(image_batch)

        # create a batch of real feature and encoded fake features
        real_and_fake_images = torch.cat([a, fake_x], dim=0).to(self.device)
        
        # create the labels for the batch
        # 1 if real, 0 if fake
        labels = torch.cat([torch.ones(a.shape[0]),torch.zeros(fake_x.shape[0])], dim=0)
        labels = labels.reshape(labels.shape[0], 1).to(self.device)
        
        # predict the labels using the discriminator
        discriminator_predictions = self.discriminator(real_and_fake_images)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(discriminator_predictions, labels)

        # return the discriminator loss, real encoded feature and real angles
        return loss, x, thetas
        
    @property
    def device(self):
        """
        Property function to get the device on which the GAN is
        """
        
        return next(self.parameters()).device


class Generator(nn.Module):
    """
	Generator part of the LeNet GAN encoder model
	"""
    
    def __init__(self, device, encoding_layer, k=2):
        """
        Generator model of the encoder

        Inputs:
            device - PyTorch device used to run the model on.
            encoding_layer - PyTorch module (e.g. sequential) representing 
                the first layers of the model in the encoder.
            k - Level of anonimity. k-1 fake features are generated
                to obscure the real feature. Default = 2
        """
        super().__init__()

        # save the inputs
        self.k = k
        self.device = device
        
        # initialize the given encoding layer
        self.encoding_layer = encoding_layer
        
    def forward(self, image_batch, training=True):
        """
        Inputs:
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            training - Boolean value indicating when training or not. Default = True
        Outputs:
            a - Real non-encoded feature. Shape: [B, C, W, H]
            x - Real encoded feature. Shape: [B, C, W, H]
            thetas - Angles used for rotating the real feature. Shape: [B, 1, 1, 1]
            fake_x - Fake generated features. Shape: [B * k-1, C, W, H]
            delta_thetas - Angles used for rotating the fake features. Shape: [B * k-1, 1, 1, 1]
        """
        
        # apply the encoding layer on the input
        a = self.encoding_layer(image_batch).to(self.device)
        
        # save the image dimensions for later use
        image_dimensions = a.shape
        
        # compute the magnitude of the image batch
        a_magnitude = torch.norm(torch.norm(a, dim=(2,3), keepdim=True), dim=1, keepdim=True)

        # create real obfuscating features b
        b = torch.normal(0, 1, size=tuple((image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3])))
        b = b.to(self.device)
        b_magnitude = torch.norm(torch.norm(b, dim=(2,3), keepdim=True), dim=1, keepdim=True)
        b = (b / b_magnitude) * a_magnitude

        # sample angles to rotate the features for the real rotation
        thetas = torch.Tensor(image_dimensions[0], 1, 1, 1).uniform_(0, 2 * np.pi).to(self.device)
        thetas = torch.exp(1j * thetas)
        
        # compute encoded real feature
        x = (a + b * 1j) * thetas
        x = x.to(self.device)
        
        # check if training
        if training:      
            # create fake obfuscating features b
            fake_a_magnitude = torch.cat([a_magnitude]*(self.k-1),dim=0)
            fake_b = torch.normal(0, 1, size=tuple(((self.k-1) * image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3]))).to(self.device)
            fake_b_magnitude = torch.norm(torch.norm(fake_b, dim=(2,3), keepdim=True), dim=1, keepdim=True)
            fake_b = (fake_b / fake_b_magnitude)* fake_a_magnitude
        
            # sample k-1 delta angles to rotate the features for fake examples
            delta_thetas = torch.Tensor((self.k-1) * image_dimensions[0], 1, 1, 1).uniform_(0, np.pi).to(self.device)
            delta_thetas = torch.exp(1j * delta_thetas)
        
            # compute encoded fake features
            fake_a = torch.cat([a]*(self.k-1),dim=0)
            fake_x = (fake_a + fake_b *1j) * delta_thetas
            fake_x = fake_x.to(self.device)
            
            # return real feature, real encoded feature, thetas, fake encoded feature and delta thetas
            return a, x, thetas, fake_x, delta_thetas
        else:
            # return the real encoded features and thetas
            return x, thetas

class Discriminator(nn.Module):
    """
	Discriminator part of the LeNet GAN encoder model
	""" 
    
    def __init__(self, device, discrim_shape):
        """
        Discriminator model of the encoder
        
        Inputs:
            device - PyTorch device used to run the model on.
            discrim_shape - Size of the discriminator input.
        """
        super().__init__()
        
        # save the inputs
        self.device = device

        # initialize the linear layer
        self.linear = nn.Linear(discrim_shape,1)
        self.discrim_shape = discrim_shape
        
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

        # cast the encoded batch to real values
        encoded_batch = encoded_batch.real

        # reshape the batch
        encoded_batch = encoded_batch.view(encoded_batch.shape[0], -1).to(self.device)
        
        # predict the labels
        predictions = self.linear(encoded_batch)
        predictions = self.sigmoid(predictions)
        
        # return the predictions
        return predictions