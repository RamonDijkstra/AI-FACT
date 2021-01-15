################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.lenet import *
from dataloaders.cifar10_loader import load_data
from models.complex_lenet_v2 import *


class ComplexLenet(pl.LightningModule):
    """
	Complex LeNet model
	"""

    def __init__(self, device, num_classes, lr,k=2):
        """
        Complex LeNet network

        Inputs:
            device - PyTorch device used to run the model on.
            k - Level of anonimity. k-1 fake features are generated
                to obscure the real feature. Default = 2
        """
        super(ComplexLenet, self).__init__()
        
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = 10
        self.k = k

        # initialize the different modules of the network
        self.encoder = GAN(self.k, device)
        self.proccessing_module = LenetProcessingModule(device)
        self.decoder = LenetDecoder(self.num_classes)

        self.loss_fn = nn.NLLLoss()

    def configure_optimizers(self):
        model_optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=0.9)
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

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, optimizer_idx)
        
        print('check2')
        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        print('check3')
        # decode the feature from
        result = self.decoder(out, thetas)
        
        print('check4')
        # return the decoded feature, discriminator predictions and real labels
        # return x, discriminator_predictions, labels
        model_loss = self.loss_fn(result, labels)
        
        print('check5')
        loss = gan_loss + model_loss

        self.log("total/loss", loss)

        return loss


class GAN(nn.Module):

    def __init__(self, k, lr):
        """
        PyTorch Lightning module that summarizes all components to train a GAN.

        Inputs:
            hidden_dims_gen  - List of hidden dimensionalities to use in the
                              layers of the generator
            hidden_dims_disc - List of hidden dimensionalities to use in the
                               layers of the discriminator
            dp_rate_gen      - Dropout probability to use in the generator
            dp_rate_disc     - Dropout probability to use in the discriminator
            z_dim            - Dimensionality of latent space
            lr               - Learning rate to use for the optimizer
        """
        super().__init__()
        # self.save_hyperparameters()
        
        self.k = k
        self.lr = lr

        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        self.generator = EncoderGenerator(k, device).to(device)
        self.discriminator = EncoderDiscriminator(device).to(device)

    def configure_optimizers(self):
        # Create optimizer for both generator and discriminator.
        # You can use the Adam optimizer for both models.
        # It is recommended to reduce the momentum (beta1) to e.g. 0.5
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        return [optimizer_gen, optimizer_disc], []

    def forward(self, x, optimizer_idx):
        """
        """
        # TODO change?
        # x, _ = batch
        print(optimizer_idx)
        if optimizer_idx == 0:
            loss, x, thetas = self.generator_step(x)
        elif optimizer_idx == 1:
            loss, x, thetas = self.discriminator_step(x)

        return loss, x, thetas

    def generator_step(self, x):
        """
        """

        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        # encode the convolved images using the generator
        with torch.no_grad():
            a, x, thetas, fake_x, delta_thetas = self.generator(x)

        batch_size = fake_x.shape[0]
        t_fake = torch.ones([batch_size,1],requires_grad=True).to(device)
        preds = self.discriminator(fake_x)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds, t_fake)

        # self.log("generator/loss", loss)

        return loss, x, thetas

    def discriminator_step(self, x):
        """
        """

        # encode the convolved images using the generator
        with torch.no_grad():
            a, x, thetas, fake_x, delta_thetas = self.generator(x)
        
        # create a batch of real feature and encoded fake features
        real_and_fake_images = torch.cat([a, fake_x], dim=0).to(self.device)
        
        # create the labels for the batch
        # 1 if real, 0 if fake
        labels = torch.cat([torch.ones(a.shape[0]),torch.zeros(fake_x.shape[0])], dim=0)
        labels = labels.reshape(labels.shape[0], 1).to(self.device)
        
        # predict the labels using the discriminator
        discriminator_predictions = self.discriminator(real_and_fake_images)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(discriminator_predictions, labels)

        # self.log("discriminator/loss", loss)

        return loss, x, thetas


def train_gan(args):
    """
    Function for training and testing a GAN model.
    The function is ready for usage. Feel free to adjust it if wanted.
    Inputs:
        args - Namespace object from the argument parser
    """

    torch.autograd.set_detect_anomaly(True)
    os.makedirs(args.log_dir, exist_ok=True)
    # load the data from the dataloader
    classes, trainloader, testloader = load_data(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Create model
    pl.seed_everything(args.seed)  # To be reproducable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    model = ComplexLenet(device, num_classes, args.lr, k=2)

    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    trainer.fit(model, trainloader)

    return model


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--model', default='MLP', type=str,
                        help='What model to use in the VAE',
                        choices=['MLP', 'CNN'])
    parser.add_argument('--hidden_dims', default=[512], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "512 256"')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=15, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='GAN_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train_gan(args)
