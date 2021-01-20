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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
from utils import *
from models.encoder.GAN import EncoderGAN

class ComplexLeNet(pl.LightningModule):
    """
	Complex LeNet model
	"""

    def __init__(self, num_classes=10, k=2, lr=3e-4):
        """
        Complex LeNet network

        Inputs:
            num_classes - Number of classes of images. Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. Default = 2
            lr - Learning rate to use for the optimizer. Default = 3e-4
        """
        super(ComplexLeNet, self).__init__()       
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr
        self.k = k

        # initialize the different modules of the network
        encoder_conv = nn.Conv2d(3, 6, 5)
        self.encoder = EncoderGAN(encoder_conv, self.k, self.lr)
        self.proccessing_module = LenetProcessingModule(self.num_classes)
        self.decoder = LenetDecoder(self.num_classes)
        self.softmax = nn.Softmax()
        
        # initialize the loss function of the complex LeNet
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """
        Function to configure the optimizers
        """
        
        # initialize optimizer for the entire model
        model_optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # return the optimizer
        return model_optimizer

    def training_step(self, batch, optimizer_idx):
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
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, optimizer_idx)

        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from
        result = self.decoder(out, thetas)

        # Log the train accuracy
        out = self.softmax(result)
        preds = out.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('train_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards
        
        # return the decoded feature, discriminator predictions and real labels
        # return x, discriminator_predictions, labels
        model_loss = self.loss_fn(result, labels)
        
        loss = gan_loss + model_loss

        # log the loss
        self.log("generator/loss", gan_loss)
        self.log("model/loss", model_loss)
        self.log("total/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, False)
        
        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from
        result = self.decoder(out, thetas)
        result = self.softmax(result)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards

class LenetProcessingModule(nn.Module):
    """
	LeNet processing module model
	"""
    
    def __init__(self, num_classes):
        """
        Processing module of the network

        Inputs:
            device - PyTorch device used to run the model on.
        """
        super(LenetProcessingModule, self).__init__()
        
        # initialize the layers of the LeNet model
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        # self.pool = nn.LPPool2d(2, 2)
        self.conv2_real = nn.Conv2d(6, 16, 5, bias=False)
        self.conv2_imag = nn.Conv2d(6, 16, 5, bias=False)                 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
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

        intermediate_real, intermediate_imag = complex_relu(encoded_batch_real, self.device), complex_relu(encoded_batch_imag, self.device)
        # intermediate_real, intermediate_imag = self.pool(intermediate_real), self.pool(intermediate_imag)
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, self.device, self.pool), complex_max_pool(intermediate_imag, self.device, self.pool)

        intermediate_real, intermediate_imag = complex_conv(intermediate_real, intermediate_imag, self.conv2_real, self.conv2_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, self.device), complex_relu(intermediate_imag, self.device)        
        # intermediate_real, intermediate_imag = self.pool(intermediate_real), self.pool(intermediate_imag)
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, self.device, self.pool), complex_max_pool(intermediate_imag, self.device, self.pool)

        intermediate_real, intermediate_imag =  intermediate_real.view(-1, 16 * 5 * 5), intermediate_imag.view(-1, 16 * 5 * 5)

        intermediate_real, intermediate_imag = self.fc1(intermediate_real), self.fc1(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, self.device), complex_relu(intermediate_imag, self.device)    

        intermediate_real, intermediate_imag = self.fc2(intermediate_real), self.fc2(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, self.device), complex_relu(intermediate_imag, self.device)

        intermediate_real, intermediate_imag = self.fc3(intermediate_real), self.fc3(intermediate_imag)    

        x = torch.complex(intermediate_real, intermediate_imag)

        return x
    
    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device

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

        self.softmax = nn.Softmax(dim=1)


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
        # decoded_batch = self.softmax(decoded_batch)
        
        # return the decoded batch
        return decoded_batch