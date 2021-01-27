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
Complex ResNet-56 model
"""

# TODO: wat hiermee?
####0 Almost all rights reserved to Philipp Lippe
### https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html

# basic imports
import os
import numpy as np 
import random
from PIL import Image
from types import SimpleNamespace

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# import utility functions (all complex functions)
from utils import *

# import encoder GAN model
from models.encoder.GAN import EncoderGAN

class ComplexResNet56(pl.LightningModule):
    """
	Complex ResNet-56 model
	"""

    def __init__(self, num_classes=10, k=2, lr=1e-3, num_blocks=[19,18,18]):
        """
        Complex LeNet network

        Inputs:
            num_classes - Number of classes of images, Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. Default = 2
            lr - Learning rate to use for the optimizer. Default = 1e-3
			num_blocks - List of ints of length 3 indicating the ResNet
				blocks per module. Default = [19,18,18] (ResNet-56)
        """
        super(ComplexResNet56, self).__init__()       
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.k = k
        self.lr = lr
        self.num_blocks = num_blocks

        # TODO: variabel maken voor ResNet-56 en ResNet-110 zeker?
        # variable to make sure the linear layer in the discriminator in the GAN has the right shape
        discriminator_linear_shape = 16*16*16

        # initialize the ResNet blocks for the encoder
        blocks = []
        blocks.append(nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False))
        blocks.append(nn.BatchNorm2d(16))
        blocks.append(nn.ReLU())
        for i in range(num_blocks[0]):
            subsample = (i == 0)
            blocks.append(
                    ResNetBlock(c_in=16,
                                             act_fn=nn.ReLU(),
                                             subsample=subsample,
                                             c_out=16)
                )
        self.input_net = nn.Sequential(*blocks)

        # initialize the different modules of the network
        self.encoder = EncoderGAN(self.input_net, self.k, self.lr)
        self.proccessing_module = Resnet_Processing_module(num_blocks[1])
        self.decoder = Resnet_Decoder(num_blocks[2], self.num_classes)
        
        # initialize the softmax for the accuracy
        self.softmax = nn.Softmax()
        
        # initialize the loss function of the complex network
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
            optimizer_idx - Int indicating the index of the current optimizer
                0 - GAN generator optimizer
                1 - GAN discriminator optimizer
                2 - Full model optimizer
        """
        
        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, optimizer_idx)
        
        # send the encoded feature to the processing module
        out = self.proccessing_module(out)
        
        # decode the feature from the processing module
        out = self.decoder(out, thetas)

        # calculate the predictions
        result = self.softmax(out)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        
        # calculate the model loss 
        model_loss = self.loss_fn(out, labels)
        loss = gan_loss + model_loss

        # log the training loss and accuracy
        self.log("train_generator_loss", gan_loss)
        self.log("train_model_loss", model_loss)
        self.log("train_total-loss", loss)
        self.log("train_acc", acc)

    def validation_step(self, batch, optimizer_idx):
        """
        Validation step of the complex LeNet model.
        
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
        
        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, False)
        
        # send the encoded feature to the processing module
        out = self.proccessing_module(out)
        
        # decode the feature from the processing module
        out = self.decoder(out, thetas)
        
        # calculate the predictions
        result = self.softmax(out)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # calculate the loss
        model_loss = self.loss_fn(out, labels)
        loss = gan_loss + model_loss
        
        # log the validation loss and accuracy
        self.log("val_generator_loss", gan_loss)
        self.log("val_model_loss", model_loss)
        self.log("val_total-loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        """
        Test step of the complex LeNet model.
        
        Inputs:
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            optimizer_idx - Int indicating the index of the current optimizer
                0 - GAN generator optimizer
                1 - GAN discriminator optimizer
                2 - Full model optimize
        """
        
        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, False)
        
        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from the processing unit
        out = self.decoder(out, thetas)
        
        # calculate the predictions
        result = self.softmax(out)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        
        # calculate the loss
        model_loss = self.loss_fn(out, labels)
        loss = gan_loss + model_loss
        
        # log the test loss and accuracy
        self.log("test_generator_loss", gan_loss)
        self.log("test_model_loss", model_loss)
        self.log("test_total-loss", loss)
        self.log("test_acc", acc)

class Resnet_Processing_module(nn.Module):
    """
    ResNet processing module model
    """

    def __init__(self,  num_blocks=18):
        """
        Processing module of the network

        Inputs:
            num_blocks - Number of ResNet blocks for this module. Default = 18
        """
        super().__init__()
        self.num_blocks = num_blocks

        # initialize the ResNet blocks
        blocks = []
        for i in range(self.num_blocks):
            subsample = (i == 0)
            blocks.append(
            ResNetBlock(c_in=32 if not subsample else 16,
                            act_fn=nn.ReLU(),
                            subsample=subsample,
                            c_out=32,
                            complex=True)
                    )
        self.blocks = nn.Sequential(*blocks)

        print(self.blocks)

    def forward(self, encoded_batch):
        """
        Forward pass of the processing module.

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

        # pass the features through the Resnet blocks
        out = self.blocks(encoded_batch)

        # return the output
        return out

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device

class Resnet_Decoder(nn.Module):
    """
    ResNet decoder model
    """

    def __init__(self, num_blocks=18, num_classes=10):
        """
        Decoder module of the network
        
        Inputs:
            num_blocks - Number of ResNet blocks for this module. Default = 18
            num_classes - Number of classes of images. Default = 10
        """
        super().__init__()
        
        # save the inputs
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        # initialize the ResNet blocks
        blocks = []
        for i in range(self.num_blocks):
            subsample = (i == 0)

            blocks.append(
                    ResNetBlock(c_in=64 if not subsample else 32,
                                    act_fn=nn.ReLU(),
                                    subsample=subsample,
                                    c_out=64)
                    )
        
        self.blocks = nn.Sequential(*blocks)
        
        # initialize the output layers 
        self.output_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(64, self.num_classes)
            )

    def forward(self,encoded_batch, thetas):
        """
        Forward pass of the decoder.
        
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
        decoded_batch = encoded_batch * torch.exp(-1j * thetas.squeeze())[:,None,None,None]

        # get rid of the imaginary part of the complex features
        decoded_batch = decoded_batch.real

        # apply the final ResNet blocks and output layers
        decoded_batch = self.blocks(decoded_batch)
        out = self.output_net(decoded_batch)
        
        # return the output
        return out

class ResNetBlock(nn.Module):
    """
    ResNet block model
    """

    def __init__(self, c_in, act_fn=nn.ReLU(), subsample=False, c_out=-1, complex=False):
        """
        Block module of the ResNet.
        
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor. Default = nn.ReLU
            subsample - If True, we want to apply a stride inside the block and reduce 
                the output shape by 2 in height and width. Default = False
            c_out - Number of output features. Note that this is only relevant if subsample 
                is True, as otherwise, c_out = c_in. Default = -1
            complex - Whether or not the block allows for complex values. Default is False
        """
        super().__init__()
        
        # save the inputs
        self.c_in = c_in
        self.act_fn = act_fn
        self.subsample = subsample
        self.c_out = c_out
        self.complex = complex

        # check whether to subsample
        if not subsample:
            c_out = c_in

        # TODO: checken of wat uitgecomment is weg kan, en text comments checken in stuk hieronder
        # check whether to use the complex version
        if self.complex:
            self.conv1_real =   nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False)
            self.conv1_imag =   nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False)
            #self.Batch_norm = nn.BatchNorm2d(c_out),
            #act_fn,
            self.conv2_real = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
            self.conv2_imag = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
            #nn.BatchNorm2d(c_out)
                  
            # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
            self.downsample_conv_real = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False) if subsample else None
            self.downsample_conv_imag = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False) if subsample else None
            #self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
            #self.act_fn = act_fn
        else:
            # network representing F
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
                nn.BatchNorm2d(c_out),
                act_fn,
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            
            # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
            self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
            self.act_fn = act_fn
        
    def forward(self, x):
        """
        Forward pass of the ResNet block.
        
        Inputs:
            x - Input batch of features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
            thetas - Angles used for rotating the real feature. Shape: [B, 1, 1, 1]
        Outputs:
            out - Output batch of features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        """
        
        # check whether to use the complex version
        if self.complex:
            # split the complex feature into its real and imaginary parts
            encoded_batch_real = x.real
            encoded_batch_imag = x.imag
            
            # pass the encoded feature through the layers
            intermediate_real, intermediate_imag = complex_conv(encoded_batch_real, encoded_batch_imag, self.conv1_real, self.conv1_imag)
            intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)

            intermediate_real, intermediate_imag = complex_relu(intermediate_real, self.device, c=1), complex_relu(intermediate_imag, self.device, c=1)

            intermediate_real, intermediate_imag = complex_conv(intermediate_real, intermediate_imag, self.conv2_real, self.conv2_imag)
            intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real), complex_batchnorm(intermediate_imag)
            
            # check whether downsampling
            if self.downsample_conv_real is not None:
                encoded_batch_real, encoded_batch_imag = complex_conv(encoded_batch_real, encoded_batch_imag, self.downsample_conv_real, self.downsample_conv_imag)
            
            # add the intermediate processed feature to the original
            intermediate_real = intermediate_real + encoded_batch_real
            intermediate_imag = intermediate_imag + encoded_batch_imag

            intermediate_real, intermediate_imag = complex_relu(intermediate_real, self.device, c=1), complex_relu(intermediate_imag, self.device, c=1)

            # recombine the real and imaginary parts into a complex feature
            out = torch.complex(intermediate_real, intermediate_imag)
            
            # return the processed complex features
            return out
        else:  
            # forward the real feature through the network
            z = self.net(x)
            
            # check whether downsampling
            if self.downsample is not None:
                x = self.downsample(x)
                
            # add the intermediate processed feature to the original
            out = z + x
            
            out = self.act_fn(out)
            
            # return the processed real features
            return out

    @property
    def device(self):
        """
        Property function to get the device on which the ResNet block is
        """
        return next(self.parameters()).device

