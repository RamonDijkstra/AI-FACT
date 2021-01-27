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
Standard LeNet model
"""

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class LeNet(pl.LightningModule):
    """
    Standard LeNet model
    """

    def __init__(self, num_classes=10, k=2, lr=3e-4):
        """
        Standard LeNet network

        Inputs:
            num_classes - Number of classes of images. Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. NOT USED HERE
            lr - Learning rate to use for the optimizer. Default = 3e-4
        """
        super(LeNet, self).__init__()
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr

        # initialize the model layers
        self.encoder_layers = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)
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
        Training step of the standard LeNet model.

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

        # run the image batch through the encoder layer
        x = self.encoder_layers(x)

        # run the feature through the rest of the layers
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        # calculate the predictions
        results = self.softmax(out)
        preds = results.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # calculate the model loss
        loss = self.loss_fn(out, labels)

        # log the training loss and accuracy
        self.log("train_model_loss", loss)
        self.log("train_total-loss", loss)
        self.log("train_acc", acc)

        # return the loss
        return loss

    def validation_step(self, batch, optimizer_idx):
        """
        Validation step of the standard LeNet model.

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

        # run the image batch through the encoder layer
        x = self.encoder_layers(x)

        # run the feature through the rest of the layers
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        # calculate the predictions
        results = self.softmax(out)
        preds = results.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # calculate the model loss
        loss = self.loss_fn(out, labels)

        # log the validation loss and accuracy
        self.log("val_model_loss", loss)
        self.log("val_total-loss", loss)
        self.log("val_acc", acc)

        # return the loss
        return loss

    def test_step(self, batch, optimizer_idx):
        """
        Test step of the standard LeNet model.

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

        # run the image batch through the encoder layer
        x = self.encoder_layers(x)

        # run the feature through the rest of the layers
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        # calculate the predictions
        results = self.softmax(out)
        preds = results.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        # calculate the model loss
        loss = self.loss_fn(out, labels)

        # log the validation loss and accuracy
        self.log("test_model_loss", loss)
        self.log("test_total-loss", loss)
        self.log("test_acc", acc)
