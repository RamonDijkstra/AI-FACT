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

class ComplexAlexNet(pl.LightningModule):
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
        super(ComplexAlexNet, self).__init__()       
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr
        self.k = k


        # initialize the different modules of the network
       
        self.model_layers = AlexNetArchitecture()
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

        # send the pictures throuygh the model
        result = self.model_layers(x)
        
        # Log the train accuracy
        preds = self.softmax(result)
        preds = preds.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('train_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards
        
        # return the decoded feature, discriminator predictions and real labels
        # return x, discriminator_predictions, labels
        model_loss = self.loss_fn(result, labels)
        
        loss = model_loss

        # log the loss
        self.log("total/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)

        # send the encoded feature to the processing unit
        result = self.model_layers(x)
        
        # decode the feature from
        preds = self.softmax(result)
        preds = preds.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards

class AlexNetArchitecture(nn.Module):

    def __init__(self):
        super(AlexNetArchitecture, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        x = x.view(x.shape[0],-1)
        #print(x.shape)
        x = self.classifier(x)
        return x