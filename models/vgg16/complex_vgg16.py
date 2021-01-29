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
Complex VGG-16 model
"""

# basic imports
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# import complex functions
from complex_functions import *

# import encoder GAN model
from models.encoder.GAN import EncoderGAN

class ComplexVGG16(pl.LightningModule):
    """
	Complex VGG-16 model
	"""

    def __init__(self, num_classes=200, k=2, lr=3e-4):
        """
        Complex VGG-16 network.

        Inputs:
            num_classes - Number of classes of images. Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. Default = 2
            lr - Learning rate to use for the optimizer. Default = 3e-4
        """
        super(ComplexVGG16, self).__init__()
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr
        self.k = k

        # initialize the number of channels
        n_channels = 3

        # initialize the model layers

        # convolution 0
        conv0 = nn.Conv2d(n_channels, 64, kernel_size = (3, 3), stride=1, padding=1)

        # preactivation 1
        preact1_batch = nn.BatchNorm2d(64)
        preact1_ReLU = nn.ReLU()
        preact1_conv = nn.Conv2d(64, 64, kernel_size = (3, 3), stride=1, padding=1)

        # convolution 1
        conv1 = nn.Conv2d(64, 128, kernel_size = (1, 1), stride=1, padding=0)

        # maxpool1
        maxpool1 = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1)

        # preactivation 2a)
        preact2a_batch = nn.BatchNorm2d(128)
        preact2a_ReLU = nn.ReLU()
        preact2a_conv = nn.Conv2d(128, 128, kernel_size = (3, 3), stride=1, padding=1)

        # preactivation 2b)
        preact2b_batch = nn.BatchNorm2d(128)
        preact2b_ReLU = nn.ReLU()
        preact2b_conv = nn.Conv2d(128, 128, kernel_size = (3, 3), stride=1, padding=1)

        # convolution 2
        conv2 = nn.Conv2d(128, 256, kernel_size = (1, 1), stride=1, padding=0)

        # maxpool2
        maxpool2 = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1)

        # preactivation 3a)
        preact3a_batch = nn.BatchNorm2d(256)
        preact3a_ReLU = nn.ReLU()
        preact3a_conv = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1)

        # preactivation 3b)
        preact3b_batch = nn.BatchNorm2d(256)
        preact3b_ReLU = nn.ReLU()
        preact3b_conv = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1)

        # create sequential for the encoder layers
        self.encoder_layers = nn.Sequential(
            conv0, preact1_batch, preact1_ReLU, preact1_conv, conv1, maxpool1, preact2a_batch, preact2a_ReLU, preact2a_conv, preact2b_batch,
            preact2b_ReLU, preact2b_conv, conv2, maxpool2, preact3a_batch, preact3a_ReLU, preact3a_conv, preact3b_batch, preact3b_ReLU,
            preact3b_conv
        )

        # initialize the different modules of the network
        self.encoder = EncoderGAN(self.encoder_layers, 50176, self.k, self.lr)
        self.proccessing_module = VGG16ProcessingModule(self.num_classes)
        self.decoder = VGG16Decoder(self.num_classes)
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
        Training step of the complex VGG-16 model.

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

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the complex VGG-16 model.

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
        Outputs:
			loss - Tensor representing the model loss.
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

        # return the loss
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step of the complex VGG-16 model.

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
        Outputs:
			loss - Tensor representing the model loss.
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

        # return the loss
        return loss

class VGG16ProcessingModule(nn.Module):
    """
	VGG-16 processing module model
	"""

    def __init__(self, num_classes=200):
        """
        Processing module of the network

        Inputs:
            num_classes - Number of classes of images. Default = 200
        """
        super(VGG16ProcessingModule, self).__init__()

        # save the inputs
        self.num_classes = num_classes

        # preactivation 3c)
        self.preact3c_conv_real = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact3c_conv_imag = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        # convolution 3
        self.conv3_real = nn.Conv2d(256, 512, kernel_size = (1, 1), stride=1, padding=0, bias=False)
        self.conv3_imag = nn.Conv2d(256, 512, kernel_size = (1, 1), stride=1, padding=0, bias=False)

        # maxpool
        self.pool = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1, return_indices=True)

        # preactivation 4a)
        self.preact4a_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact4a_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        # preactivation 4b)
        self.preact4b_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact4b_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        # preactivation 4c)
        self.preact4c_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact4c_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        # preactivation 5a)
        self.preact5a_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact5a_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        # preactivation 5b)
        self.preact5b_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact5b_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

        # preactivation 5c)
        self.preact5c_conv_real = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)
        self.preact5c_conv_imag = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1, bias=False)

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

        # split the complex feature into its real and imaginary parts
        encoded_batch_real = encoded_batch.real
        encoded_batch_imag = encoded_batch.imag

        # pass the encoded feature through the layers

        # preact 3c
        intermediate_real, intermediate_imag = complex_batchnorm(encoded_batch_real, encoded_batch_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact3c_conv_real, self.preact3c_conv_imag
        )

        # conv 3
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.conv3_real, self.conv3_imag
        )

        # pool 3
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)

        # preact 4a
        intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact4a_conv_real, self.preact4a_conv_imag
        )

        # preact 4b
        intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact4b_conv_real, self.preact4b_conv_imag
        )

        # preact 4c
        intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact4c_conv_real, self.preact4c_conv_imag
        )

        # pool 4
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)


        # preact 5a
        intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact5a_conv_real, self.preact5a_conv_imag
        )

        # preact 5b
        intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact5b_conv_real, self.preact5b_conv_imag
        )

        # preact 5c
        intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_conv(
            intermediate_real, intermediate_imag, self.preact5c_conv_real, self.preact5c_conv_imag
        )

        # pool 5
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)

        # last batchnorm
        intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, intermediate_imag)

        # last ReLU
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)

        # recombine the real and imaginary parts into a complex feature
        processed_batch = torch.complex(intermediate_real, intermediate_imag)

        # return the processed complex features
        return processed_batch

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device

class VGG16Decoder(nn.Module):
    """
	VGG-16 decoder model
	"""

    def __init__(self, num_classes=200):
        """
        Decoder module of the network

		Inputs:
            num_classes - Number of classes of images. Default = 200
        """
        super(VGG16Decoder, self).__init__()

        # save the inputs
        self.num_classes = num_classes

        # initialize the layers
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoded_batch, thetas):
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

        # get the real part of the complex features
        decoded_batch = encoded_batch.real

        # reshape the batch
        decoded_batch = decoded_batch.reshape(decoded_batch.shape[0], -1)

        # pass the decoded batch through the layers
        decoded_batch = self.fc1(decoded_batch)
        decoded_batch = self.fc2(decoded_batch)
        decoded_batch = self.fc3(decoded_batch)

        # return the decoded batch
        return decoded_batch

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device
