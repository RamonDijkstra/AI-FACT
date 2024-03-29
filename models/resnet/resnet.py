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
Standard ResNet model
"""

# basic imports
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ResNet(pl.LightningModule):
    """
    ResNet model
    """

    def __init__(self, num_classes=10, k=2, lr=3e-4, num_blocks=[19,18,18]):
        """
        ResNet network

        Inputs:
            num_classes - Number of classes of images. Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. NOT USED HERE
            lr - Learning rate to use for the optimizer. Default = 3e-4
            num_blocks - Number of blocks per module. Default = [19,18,18]
                (ResNet-56)
        """
        super(ResNet, self).__init__()
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr
        self.k = k
        self.num_blocks = num_blocks

        # initialize the ResNet blocks for the encoder layers
        blocks = []
        blocks.append(nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False))
        blocks.append(nn.BatchNorm2d(16))
        blocks.append(nn.ReLU())
        for i in range(num_blocks[0]):
            subsample = (i == 0)
            blocks.append(
                ResNetBlock(
                    c_in=16,
                    act_fn=nn.ReLU(),
                    subsample=subsample,
                    c_out=16
                )
            )
        self.encoder_layers = nn.Sequential(*blocks)

        # initialize the rest of the ResNet blocks
        blocks = []
        for i in range(self.num_blocks[1]):
            subsample = (i == 0)
            blocks.append(
                ResNetBlock(
                    c_in=32 if not subsample else 16,
                    act_fn=nn.ReLU(),
                    subsample=subsample,
                    c_out=32
                )
            )
        for i in range(self.num_blocks[2]):
            subsample = (i == 0)
            blocks.append(
                ResNetBlock(
                    c_in=64 if not subsample else 32,
                    act_fn=nn.ReLU(),
                    subsample=subsample,
                    c_out=64
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # initialize the output layers
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, self.num_classes)
        )

        # softmax is used to obtain the accuracy
        self.softmax = nn.Softmax(dim=1)

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
        Training step of the standard ResNet model.

        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            optimizer_idx - Int indicating the index of the current optimizer
                0 - Full model optimizer
        Outputs:
			loss - Tensor representing the model loss.
        """

        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the network
        x = self.encoder_layers(x)
        x = self.blocks(x)
        out = self.output_net(x)

        # calculate the predictions
        results = self.softmax(out)
        preds = results.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # calculate the model loss
        loss = self.loss_fn(out, labels)

        # log the training loss and accuracy
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        # return the loss
        return loss

    def validation_step(self, batch, optimizer_idx):
        """
        Validation step of the standard ResNet model.

        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            optimizer_idx - Int indicating the index of the current optimizer
                0 - Full model optimizer
        Outputs:
			loss - Tensor representing the model loss.
        """

        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the network
        x = self.encoder_layers(x)
        x = self.blocks(x)
        out = self.output_net(x)

        # calculate the predictions
        results = self.softmax(out)
        preds = results.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # calculate the model loss
        loss = self.loss_fn(out, labels)

        # log the validation loss and accuracy
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        # return the loss
        return loss

    def test_step(self, batch, optimizer_idx):
        """
        Test step of the standard ResNet model.

        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            optimizer_idx - Int indicating the index of the current optimizer
                0 - Full model optimizer
        Outputs:
			loss - Tensor representing the model loss.
        """

        # divide the batch in images and labels
        x, labels = batch

        # run the image batch through the network
        x = self.encoder_layers(x)
        x = self.blocks(x)
        out = self.output_net(x)

        # calculate the predictions
        results = self.softmax(out)
        preds = results.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # calculate the model loss
        loss = self.loss_fn(out, labels)

        # log the test loss and accuracy
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        # return the loss
        return loss

class ResNetBlock(nn.Module):
    """
    ResNet block
    """

    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Class for ResNet block.

        Inputs:
            c_in - Int indicating the number of input features.
            act_fn - Activation class constructor (e.g. nn.ReLU).
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()

        # if not subsampling, the output is the same as the input
        if not subsample:
            c_out = c_in

        # a ResNet block
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
