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
        self.encoder_layers = nn.Conv2d(3, 6, 5)
        self.encoder = EncoderGAN(self.encoder_layers, (6*28*28), self.k, self.lr)
        self.proccessing_module = LenetProcessingModule(self.num_classes)
        self.decoder = LenetDecoder(self.num_classes)
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
        Training step of the complex LeNet model.

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

    def test_step(self, batch, optimizer_idx):
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

class LenetProcessingModule(nn.Module):
    """
	LeNet processing module model
	"""

    def __init__(self, num_classes=10):
        """
        Processing module of the network

        Inputs:
            num_classes - Number of classes of images. Default = 10
        """
        super(LenetProcessingModule, self).__init__()

        # initialize the layers of the LeNet model
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2_real = nn.Conv2d(6, 16, 5, bias=False)
        self.conv2_imag = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

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
        intermediate_real, intermediate_imag = complex_relu(encoded_batch_real, encoded_batch_imag, self.device)
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)

        intermediate_real, intermediate_imag = complex_conv(intermediate_real, intermediate_imag, self.conv2_real, self.conv2_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)
        intermediate_real, intermediate_imag = complex_max_pool(intermediate_real, intermediate_imag, self.pool)

        intermediate_real, intermediate_imag =  intermediate_real.view(-1, 16 * 5 * 5), intermediate_imag.view(-1, 16 * 5 * 5)

        intermediate_real, intermediate_imag = self.fc1(intermediate_real), self.fc1(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)

        intermediate_real, intermediate_imag = self.fc2(intermediate_real), self.fc2(intermediate_imag)
        intermediate_real, intermediate_imag = complex_relu(intermediate_real, intermediate_imag, self.device)

        intermediate_real, intermediate_imag = self.fc3(intermediate_real), self.fc3(intermediate_imag)

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

class LenetDecoder(nn.Module):
    """
	LeNet decoder model
	"""

    def __init__(self, num_classes):
        """
        Decoder module of the network

		Inputs:
            num_classes - Number of classes of images. Default = 10
        """
        super(LenetDecoder, self).__init__()

        # save the inputs
        self.num_classes = num_classes


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
        decoded_batch = encoded_batch * torch.exp(-1j * thetas.squeeze())[:, None]

        # get the real part of the complex features
        decoded_batch = decoded_batch.real

        # return the decoded batch
        return decoded_batch
