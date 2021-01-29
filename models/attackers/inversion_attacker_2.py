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
Model for inversion attacks (inversion attacker 2)
"""

# basic imports
import argparse
import os

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl

class UNet(pl.LightningModule):
    """
    Standard U-net model
    """

    def __init__(self, generator=None, encoding_layer=None, enc_chs=(6,64,128,256,512), dec_chs=(512, 256, 128, 64), num_channels=3, retain_dim=True, out_sz=(32,32), lr=3e-4):
        """
        Standard U-net network

        Inputs:
            generator - GAN class object.
            encoding_layer - Layers of the encoder before the GAN.
            enc_chs - Channels of the U-net downsampling blocks.
                Default = (6,64,128,256,512)
            dec_chs - Channels of the U-net upsampling blocks.
                Default = (512, 256, 128, 64)
            num_channels - Int indicating the number of channels of the input images. Default = 3
            retain_dim - Boolean indicating whether to retain the input dimension. Default = True
            out_sz - Tuple indicating the shape of the output size. Default = (32,32)
            lr - Learning rate to use for the optimizer. Default = 3e-4
        """
        super().__init__()

        # save the inputs
        self.gan = generator
        self.encoding_layer = encoding_layer
        self.out_sz = out_sz
        self.retain_dim = retain_dim
        self.lr = lr

        # initialize first upsampling layer for the features
        self.upsample = nn.Upsample(size=(32,32))

        # check whether the GAN is initialized
        if self.gan is not None:
            self.gan.requires_grad = False

        # check whether the encoding layers are initialized
        if self.encoding_layer is not None:
            self.encoding_layer.requires_grad = False

        # initialize the model layers
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_channels, 1)

        # initialize the loss function
        self.loss_fn = nn.MSELoss(reduction='mean')

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
        Training step of the standard U-net model.

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

        # get the images of the batch
        x, _ = batch

        # run the image batch through the encoding layer
        with torch.no_grad():
            x2 = self.encoding_layer(x)

        # run the batch through the layers
        enc_ftrs = self.encoder(x2)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)

        # calculate the model loss
        loss = self.loss_fn(out, x)

        # log the training loss
        self.log("train_loss", loss)

        # return the loss
        return loss

    def validation_step(self, batch, optimizer_idx):
        """
        Validation step of the standard U-net model.

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

        # get the images of the batch
        x, _ = batch

        # run the image batch through the encoding layer
        with torch.no_grad():
            x2 = self.encoding_layer(x)

        # run the batch through the layers
        enc_ftrs = self.encoder(x2)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)

        # calculate the model loss
        loss = self.loss_fn(out, x)

        # log the validation loss
        self.log("val_loss", loss)

        # return the loss
        return loss


    def test_step(self, batch, optimizer_idx):
        """
        Test step of the standard U-net model.

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

        # get the images of the batch
        x, _ = batch

        # run the image batch through the encoding layer
        with torch.no_grad():
            x2 = self.encoding_layer(x)

        # run the image batch through the GAN
        _, encoded_x, thetas, _, _ = self.gan(x)

        # upsample both processed batches
        encoded_x = self.upsample(encoded_x.real)
        encoded_x2 = self.upsample(x2)

        # run the batches through the layers
        enc_ftrs = self.encoder(encoded_x)
        enc_ftrs2 = self.encoder(encoded_x2)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out2 = self.decoder(enc_ftrs2[::-1][0], enc_ftrs2[::-1][1:])
        out = self.head(out)
        out2 = self.head(out2)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
            out2 = F.interpolate(out2, self.out_sz)

        # calculate the model loss of the obfuscated features
        obfuscated_loss = self.loss_fn(out, x)

        # calculate the model loss of the non-obfuscated features
        non_obfuscated_loss = self.loss_fn(out2,x)

        # log the losses
        self.log("Obfuscated - reconstruction_error", obfuscated_loss)
        self.log("Non-obfuscated - reconstruction_error", non_obfuscated_loss)

        # return the losses
        return obfuscated_loss, non_obfuscated_loss

class Encoder(nn.Module):
    """
    U-net encoder half
    """

    def __init__(self, chs=(6,64,128,256,512,1024)):
        """
        Class for U-net encoder half.

        Inputs:
            chs - The channels of the block of the encoder.
                Default = (6,64,128,256,512,1024)
        """
        super().__init__()

        # initialize the layers of the U-net encoder
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """
        Forward of the U-net encoder.

        Inputs:
            x - Input batch of features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        Outputs:
			ftrs - Batch of compressed features.
        """

        # forward the batch through the Layers
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)

        # return the compressed features
        return ftrs

class Decoder(nn.Module):
    """
    U-net decoder half
    """

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        """
        Class for U-net decoder half.

        Inputs:
            chs - The channels of the block of the encoder.
                Default = (1024, 512, 256, 128, 64)
        """
        super().__init__()

        # save the inputs
        self.chs = chs

        # initialize the layers of the U-net decoder
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        """
        Forward of the U-net decoder.

        Inputs:
            x - Input batch of features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
            encoder_features - Batch of compressed features from the encoder.
        Outputs:
			x - Batch of decoded features.
        """

        # forward the batch through the Layers
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)

        # return the decoded features
        return x

    def crop(self, enc_ftrs, x):
        """
        Crop function of the U-net decoder.

        Inputs:
            enc_ftrs - Batch of compressed features from the encoder.
            x - Input batch of features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        Outputs:
			enc_features - Batch of cropped features.
        """

        # crop the features
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)

        # return the cropped features
        return enc_ftrs

class Block(nn.Module):
    """
    U-net block
    """

    def __init__(self, in_ch, out_ch):
        """
        Class for U-net block.

        Inputs:
            in_ch - Int indicating the number of input channels.
            out_ch - Int indicating the number of output channels.
        """
        super().__init__()

        # initialize the layers of the U-net block
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )

    def forward(self, x):
        """
        Forward of the U-net block.

        Inputs:
            x - Input batch of features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        Outputs:
			out - Processed batch.
        """

        # the batch through the Layers
        out = self.layers(x)

        # return the output
        return out
