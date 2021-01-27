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

class Complex_VGG16(pl.LightningModule):
    """
	Complex VGG16 model
	"""

    def __init__(self, num_classes=10, k=2, lr=3e-4):
        """
        Standard VGG16 network

        Inputs:
            num_classes - Number of classes of images. Default = 10
            lr - Learning rate to use for the optimizer. Default = 3e-4
        """
        super(Complex_VGG16, self).__init__()
        self.save_hyperparameters()
        
        # save the inputs
        self.num_classes = num_classes
        self.lr = lr
        self.k = k

        n_channels = 3

        # initialize the model layers

        #Convolution 0
        conv0 = nn.Conv2d(n_channels, 64, kernel_size = (3, 3), stride=1, padding=1)

        #Preactivation 1
        preact1_batch = nn.BatchNorm2d(64)
        preact1_ReLU = nn.ReLU()
        preact1_conv = nn.Conv2d(64, 64, kernel_size = (3, 3), stride=1, padding=1)

        #Convolution 1
        conv1 = nn.Conv2d(64, 128, kernel_size = (1, 1), stride=1, padding=0)

        #Maxpool1
        maxpool1 = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1)

        #Preactivation 2a)
        preact2a_batch = nn.BatchNorm2d(128)
        preact2a_ReLU = nn.ReLU()
        preact2a_conv = nn.Conv2d(128, 128, kernel_size = (3, 3), stride=1, padding=1)

        #Preactivation 2b)
        preact2b_batch = nn.BatchNorm2d(128)
        preact2b_ReLU = nn.ReLU()
        preact2b_conv = nn.Conv2d(128, 128, kernel_size = (3, 3), stride=1, padding=1)

        #Convolution 2
        conv2 = nn.Conv2d(128, 256, kernel_size = (1, 1), stride=1, padding=0)

        #Maxpool2
        maxpool2 = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1)

        #Preactivation 3a)
        preact3a_batch = nn.BatchNorm2d(256)
        preact3a_ReLU = nn.ReLU()
        preact3a_conv = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1)

        #Preactivation 3b)
        preact3b_batch = nn.BatchNorm2d(256)
        preact3b_ReLU = nn.ReLU()
        preact3b_conv = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1)

        # self.input_net = nn.Sequential(
        #     conv0, preact1_batch, preact1_ReLU, preact1_conv, conv1, maxpool1, preact2a_batch, preact2a_ReLU, preact2a_conv, preact2b_batch,
        #     preact2b_ReLU, preact2b_conv, conv2, maxpool2, preact3a_batch, preact3a_ReLU, preact3a_conv, preact3b_batch, preact3b_ReLU,
        #     preact3b_conv
        # )

        self.input_net = nn.Sequential(
            conv0, preact1_ReLU, preact1_conv, conv1, maxpool1, preact2a_ReLU, preact2a_conv,
            preact2b_ReLU, preact2b_conv, conv2, maxpool2, preact3a_ReLU, preact3a_conv, preact3b_ReLU,
            preact3b_conv
        )

        #number depended on dataset
        self.encoder = EncoderGAN(self.input_net, (256*8*8), self.k, self.lr)
        self.proccessing_module = VGG16ProcessingModule(self.num_classes)
        self.decoder = VGG16Decoder(self.num_classes)
        self.softmax = nn.Softmax()
        
        # initialize the loss function
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
        Training step of the complex LeNet model.
        
        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
        Outputs:
			decoded_feature - Output batch of decoded real features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
            discriminator_predictions - Predictions from the discriminator. Shape: [B * k, 1]
            labels - Real labels of the encoded features. Shape: [B * k, 1]
        """
        
        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, optimizer_idx)

        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from
        result = self.decoder(out, thetas)

        # log the train accuracy
        out = self.softmax(result)
        preds = out.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('train_acc', acc)
        
        # return the decoded feature, discriminator predictions and real labels
        # return x, discriminator_predictions, labels
        model_loss = self.loss_fn(result, labels)
        loss = gan_loss + model_loss

        # log the loss
        self.log("generator/loss", gan_loss)
        self.log("model/loss", model_loss)
        self.log("total/loss", loss)

        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the complex LeNet model.
        
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
        
        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, False)
        
        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from the processing unit
        result = self.decoder(out, thetas)
        
        # calculate the predictions
        result = self.softmax(result)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        model_loss = self.loss_fn(result, labels)
        loss = gan_loss + model_loss
        
        # log the validation accuracy
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        """
        Test step of the complex LeNet model.
        
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
        
        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, False)
        
        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from the processing unit
        result = self.decoder(out, thetas)
        
        # calculate the predictions
        result = self.softmax(result)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        
        # log the test accuracy
        self.log('test_acc', acc)

class VGG16ProcessingModule(nn.Module):
    """
	VGG16 processing module model
	"""
    
    def __init__(self, num_classes):
        """
        Processing module of the network

        Inputs:
            device - PyTorch device used to run the model on.
        """
        super(VGG16ProcessingModule, self).__init__()

        self.num_classes = num_classes

        #Preactivation 3c)
        self.preact3c_conv_real = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact3c_conv_imag = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        #Convolution 3
        self.conv3_real = nn.Conv2d(256, 512, kernel_size = (1, 1), stride=1, padding=0, bias=False)
        self.conv3_imag = nn.Conv2d(256, 512, kernel_size = (1, 1), stride=1, padding=0, bias=False)

        #Maxpool
        self.pool = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1, return_indices=True)

        #Preactivation 4a)
        self.preact4a_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact4a_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        #Preactivation 4b)
        self.preact4b_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact4b_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        #Preactivation 4c)
        self.preact4c_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact4c_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        #Preactivation 5a)
        self.preact5a_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact5a_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        #Preactivation 5b)
        self.preact5b_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact5b_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        #Preactivation 5c)
        self.preact5c_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact5c_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
    
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

        # Preact 3c
        # intermediate_real, intermediate_imag = complex_batchnorm(encoded_batch_real), complex_batchnorm(encoded_batch_imag)
        intermediate_real, intermediate_imag = complex_relu(encoded_batch_real, encoded_batch_imag, self.device)
        # intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact3c_conv_real, self.preact3c_conv_imag
        )

        # Conv 3
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.conv3_real, self.conv3_imag
        )

        # Pool 3
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)

        # Preact 4a
        # intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact4a_conv_real, self.preact4a_conv_imag
        )

        # Preact 4b
        # intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact4b_conv_real, self.preact4b_conv_imag
        )

        # Preact 4c
        # intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact4c_conv_real, self.preact4c_conv_imag
        )

        # Pool 4
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)
        

        # Preact 5a
        # intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact5a_conv_real, self.preact5a_conv_imag
        )

        # Preact 5b
        # intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact5b_conv_real, self.preact5b_conv_imag
        )

        # Preact 5c
        # intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact5c_conv_real, self.preact5c_conv_imag
        )

        # Pool 5
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)

        # Last batchnorm
        # intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
        
        # Last ReLU
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)

        x = torch.complex(intermediate_real, intermediate_imag)

        return x
    
    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device

class VGG16Decoder(nn.Module):
    """
	VGG16 decoder model
	"""
    
    def __init__(self, num_classes):
        """
        Decoder module of the network
        """
        super(VGG16Decoder, self).__init__()

        # initialize the softmax layer
        self.num_classes = num_classes
        
        # self.fc1 = nn.Linear(16384, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

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
        # decoded_batch = encoded_batch * torch.exp(-1j * thetas.squeeze())[:,None,None,None]

        # get rid of the imaginary part of the complex features
        # decoded_batch = decoded_batch.real
        decoded_batch = encoded_batch.real

        decoded_batch = decoded_batch.reshape(decoded_batch.shape[0], -1)

        self.fc1 = nn.Linear(decoded_batch.shape[1], 4096).to(self.device)

        decoded_batch = self.fc1(decoded_batch)
        decoded_batch = self.fc2(decoded_batch)
        result = self.fc3(decoded_batch)
        
        # apply the softmax layer#
        # decoded_batch = self.softmax(decoded_batch)
        
        # return the decoded batch
        return result

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device