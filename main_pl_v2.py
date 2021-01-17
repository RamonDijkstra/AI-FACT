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
    
    Inputs:
        args - Namespace object from the argument parser
    """
    
    # DEBUG
    torch.autograd.set_detect_anomaly(True)
    
    # make folder for the Lightning logs
    os.makedirs(args.log_dir, exist_ok=True)\
    
    # load the data from the dataloader
    classes, trainloader, testloader = load_data(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # initialize the Lightning trainer
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    trainer.logger._default_hp_metric = None

    # seed for reproducability
    pl.seed_everything(args.seed)
    
    # initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO: make models variable depending on command line arguments
    num_classes = len(classes)
    model = ComplexLenet(num_classes=num_classes, lr=args.lr, k=args.k)

    # show the progress bar if enabled
    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # train the model
    trainer.fit(model, trainloader)

    # test the model
    trainer.test(test_dataloaders=testloader)

    # return the model
    return model


if __name__ == '__main__':
    """
    Direct calling of the python file via command line.
    Handles the given hyperparameters
    """
    
    # initialize the parser for the command line arguments
    parser = argparse.ArgumentParser()
    
    # --- OUR HYPERPARAMETERS ---
    
    # model hyperparameters
    parser.add_argument('--model', default='LeNet', type=str,
                        help='What complex model to use. Default is LeNet.',
                        choices=['LeNet', 'ResNet'])
    
    # dataloader hyperparameters
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Minibatch size. Default is 4.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. Default is 0 (truly deterministic).')
                        
    # optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use. Default is 1e-3.')
                        
    # training hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs. Default is 10.')
    parser.add_argument('--k', default=2, type=int,
                        help='Level of anonimity to use during training. k-1 fake features are generated to train the encoder. Default is 2,')                   
    parser.add_argument('--log_dir', default='GAN_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs are created. Default is GAN_logs/.')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator. Disabled by default.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results. Default is 42.')
    
    # --- OLD HYPERPARAMETERS ---

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[512], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "512 256"')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # -------------------------
    
    # parse the arguments 
    args = parser.parse_args()

    # train the model with the given arguments
    train_model(args)
