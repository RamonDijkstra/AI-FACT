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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.lenet import *
from dataloaders.cifar10_loader import load_data
from models.complex_lenet_v3 import *


def train_model(args):
    """
    Function for training and testing a model.
    The function is ready for usage. Feel free to adjust it if wanted.
    Inputs:
        args - Namespace object from the argument parser
    """
    
    # DEBUG
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
    model = ComplexLenet(num_classes, args.lr, k=2)

    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    trainer.fit(model, trainloader)

    trainer.test(test_dataloaders=testloader)

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
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=2, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='GAN_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train_model(args)
