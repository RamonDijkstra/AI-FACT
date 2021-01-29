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
Complex ResNet model
"""

# basic imports
import os
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# import complex functions
from complex_functions import *

# import encoder GAN model
from models.encoder.GAN import EncoderGAN

class ComplexResNet(pl.LightningModule):
    """
	Complex ResNet model
	"""

    def __init__(self, num_classes=10, k=2, lr=1e-3, num_blocks=[19,18,18]):
        """
        Complex ResNet network

        Inputs:
            num_classes - Number of classes of images, Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. Default = 2
            lr - Learning rate to use for the optimizer. Default = 1e-3
            num_blocks - Number of blocks per module. Default = [19,18,18]
                (ResNet-56)
        """
        super(ComplexResNet, self).__init__()
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr
        self.k = k

        # variable to make sure the linear layer in the discriminator in the GAN has the right shape
        discriminator_linear_shape = 16*16*16

        # initialize the different modules of the network
        # create the ResNet blocks of the encoder and initialize the encoder
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
        self.encoder_layers = nn.Sequential(*blocks)
        self.encoder = EncoderGAN(self.encoder_layers, discriminator_linear_shape, self.k, self.lr)

        # initialize the processing module and decoder
        self.proccessing_module = Resnet_Processing_module(num_blocks[1])
        self.decoder = Resnet_Decoder(num_blocks[2], self.num_classes)

        # initialize the softmax
        self.softmax = nn.Softmax(dim=1)

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
        Training step of the complex ResNet model.

        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
			optimizer_idx - Int indicating the index of the current optimizer
                0 - GAN generator optimizer
                1 - GAN discriminator optimizer
                2 - Full model optimizer
        Outputs:
			loss - Tensor representing the model loss.
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

        # return the loss
        return loss

    def validation_step(self, batch, optimizer_idx):
        """
        Validation step of the complex ResNet model.

        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
			optimizer_idx - Int indicating the index of the current optimizer
                0 - GAN generator optimizer
                1 - GAN discriminator optimizer
                2 - Full model optimizer
        Outputs:
			loss - Tensor representing the model loss.
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

        # log the validation loss and accuracy
        self.log("val_generator_loss", gan_loss)
        self.log("val_model_loss", model_loss)
        self.log("val_total-loss", loss)
        self.log("val_acc", acc)

        # return the loss
        return loss

    def test_step(self, batch, optimizer_idx):
        """
        Test step of the complex ResNet model.

        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
			optimizer_idx - Int indicating the index of the current optimizer
                0 - GAN generator optimizer
                1 - GAN discriminator optimizer
                2 - Full model optimizer
        Outputs:
			loss - Tensor representing the model loss.
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

        # log the test loss and accuracy
        self.log("test_generator_loss", gan_loss)
        self.log("test_model_loss", model_loss)
        self.log("test_total-loss", loss)
        self.log("test_acc", acc)

        # return the loss
        return loss

class Resnet_Processing_module(nn.Module):
    """
	ResNet processing module model
	"""

    def __init__(self,  num_blocks):
        """
        Processing module of the network

        Inputs:
            num_block - Number of ResNet blocks.
        """
        super().__init__()

        # save the inputs
        self.num_blocks = num_blocks

        # create the ResNet blocks for the processing module
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

        # pass the encoded feature through the layers
        processed_batch = self.blocks(encoded_batch)

        # return the processed complex features
        return processed_batch

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

    def __init__(self, num_blocks, num_classes):
        """
        Decoder module of the network

		Inputs:
            num_block - Number of ResNet blocks.
            num_classes - Number of classes of images.
        """
        super().__init__()

        # save the inputs
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        # create the ResNet blocks for the decoder
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

        # create the final layers for the decoder
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
        decoded_batch = encoded_batch * torch.exp(-1j * thetas.squeeze())[:, None, None, None]

        # get the real part of the complex features
        decoded_batch = decoded_batch.real

        # pass the decoded feature through the layers
        decoded_batch = self.blocks(decoded_batch)
        decoded_batch = self.output_net(decoded_batch)

        # return the decoded batch
        return decoded_batch

class ResNetBlock(nn.Module):
    """
    ResNet block
    """

    def __init__(self, c_in, act_fn=nn.ReLU(), subsample=False, c_out=-1, complex=False):
        """
        Class for ResNet block.

        Inputs:
            c_in - Int indicating the number of input features.
            act_fn - Activation class constructor (e.g. nn.ReLU).
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
            complex - Boolean indicating whether to create a complex or normal block
        """
        super().__init__()

        # save the inputs
        self.complex = complex

        # if not subsampling, the output is the same as the input
        if not subsample:
            c_out = c_in

        # a complex ResNet block
        if self.complex:
            self.Conv1_real =   nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False)
            self.Conv1_imag =   nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False)
            self.Conv2_real = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
            self.Conv2_imag = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)

            # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
            self.downsample_conv_real = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False) if subsample else None
            self.downsample_conv_imag = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False) if subsample else None
        # a ResNet block
        else:
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False), # No bias needed as the Batch Norm handles it
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
        Forward of the ResNet block.

        Inputs:
            x - Input batch of features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        Outputs:
			out - Processed batch. Either downsampled or not
        """

        # check whether using the complex block
        if self.complex:
            # split the complex feature into its real and imaginary parts
            x_real = x.real
            x_imag = x.imag

            # pass the encoded feature through the layers
            intermediate_real, intermediate_imag = complex_conv(x_real, x_imag, self.Conv1_real, self.Conv1_imag)
            intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)

            intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)

            intermediate_real, intermediate_imag = complex_conv(intermediate_real, intermediate_imag, self.Conv2_real, self.Conv2_imag)
            intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)

            # check if downsampling
            if self.downsample_conv_real is not None:
                x_real, x_imag = complex_conv(x_real, x_imag, self.downsample_conv_real, self.downsample_conv_imag)

            # add the input to the output
            intermediate_real = intermediate_real + x_real
            intermediate_imag = intermediate_imag + x_imag
            intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)

            # recombine the real and imaginary parts into a complex feature
            out = torch.complex(intermediate_real, intermediate_imag)

            # return the processed features
            return out
        else:
            # pass the input through the ResNet block
            z = self.net(x)

            # check if downsampling
            if self.downsample is not None:
                x = self.downsample(x)

            # add the input to the output
            out = z + x
            out = self.act_fn(out)

            # return the processed features
            return out

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device
