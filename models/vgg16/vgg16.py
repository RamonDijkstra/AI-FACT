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
Standard VGG-16 model
"""

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class VGG16(pl.LightningModule):
    """
    Standard VGG-16 model
    """

    def __init__(self, num_classes=10, k=2, lr=3e-4):
    # def __init__(self, n_channels, n_classes):
        """
        Standard VGG-16 network.

        Inputs:
            num_classes - Number of classes of images. Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. NOT USED HERE
            lr - Learning rate to use for the optimizer. Default = 3e-4
        """
        super(VGG16, self).__init__()
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr

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

        # preactivation 3c)
        preact3c_batch = nn.BatchNorm2d(256)
        preact3c_ReLU = nn.ReLU()
        preact3c_conv = nn.Conv2d(256, 256, kernel_size = (3, 3), stride=1, padding=1)

        # convolution 3
        conv3 = nn.Conv2d(256, 512, kernel_size = (1, 1), stride=1, padding=0)

        # maxpool3
        maxpool3 = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1)

        # preactivation 4a)
        preact4a_batch = nn.BatchNorm2d(512)
        preact4a_ReLU = nn.ReLU()
        preact4a_conv = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1)

        # preactivation 4b)
        preact4b_batch = nn.BatchNorm2d(512)
        preact4b_ReLU = nn.ReLU()
        preact4b_conv = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1)

        # preactivation 4c)
        preact4c_batch = nn.BatchNorm2d(512)
        preact4c_ReLU = nn.ReLU()
        preact4c_conv = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1)

        # maxpool4
        maxpool4 = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1)

        # preactivation 5a)
        preact5a_batch = nn.BatchNorm2d(512)
        preact5a_ReLU = nn.ReLU()
        preact5a_conv = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1)

        # preactivation 5b)
        preact5b_batch = nn.BatchNorm2d(512)
        preact5b_ReLU = nn.ReLU()
        preact5b_conv = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1)

        # preactivation 5c)
        preact5c_batch = nn.BatchNorm2d(512)
        preact5c_ReLU = nn.ReLU()
        preact5c_conv = nn.Conv2d(512, 512, kernel_size = (3, 3), stride=1, padding=1)

        # maxpool5
        maxpool5 = nn.MaxPool2d(kernel_size = (3, 3), stride=2, padding=1)

        # batchnormlayer and ReLU activation function
        last_batch_layer = nn.BatchNorm2d(512)
        last_ReLU_layer = nn.ReLU()

        # create sequential for the encoder layers
        self.encoder_layers = nn.Sequential(
            conv0, preact1_batch, preact1_ReLU, preact1_conv, conv1, maxpool1, preact2a_batch, preact2a_ReLU, preact2a_conv, preact2b_batch,
            preact2b_ReLU, preact2b_conv, conv2, maxpool2, preact3a_batch, preact3a_ReLU, preact3a_conv, preact3b_batch, preact3b_ReLU,
            preact3b_conv
        )

        # create sequential for the other layers
        self.layers = nn.Sequential(
            preact3c_batch, preact3c_ReLU, preact3c_conv, conv3, maxpool3, preact4a_batch, preact4a_ReLU, preact4a_conv,
            preact4b_batch, preact4b_ReLU, preact4b_conv, preact4c_batch, preact4c_ReLU, preact4c_conv, maxpool4, preact5a_batch,
            preact5a_ReLU, preact5a_conv, preact5b_batch, preact5b_ReLU, preact5b_conv, preact5c_batch, preact5c_ReLU, preact5c_conv,
            maxpool5, last_batch_layer, last_ReLU_layer
        )

        # linear layers from original paper
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

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
        Training step of the standard VGG-16 model.

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
        x = self.layers(x)

        # reshape the features
        x = x.reshape(x.shape[0], -1)

        # pass through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

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

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the standard VGG-16 model.

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
        x = self.layers(x)

        # reshape the features
        x = x.reshape(x.shape[0], -1)

        # pass through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

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

    def test_step(self, batch, batch_idx):
        """
        Test step of the standard VGG-16 model.

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
        x = self.layers(x)

        # reshape the features
        x = x.reshape(x.shape[0], -1)

        # pass through the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

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
