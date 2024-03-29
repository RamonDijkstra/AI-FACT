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
Main training for attacker file
"""

# basic imports
import argparse
import os
from os import listdir
from os.path import isfile, join

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# import models
from models.attackers.inversion_attacker_2 import *
from models.lenet.complex_lenet import *
from models.resnet.complex_resnet import *

# import dataloaders
from dataloaders.cifar10_loader import load_data as load_cifar10_data
from dataloaders.cifar100_loader import load_data as load_cifar100_data

# initialize our model dictionary
gan_model_dict = {}
gan_model_dict['Complex_LeNet'] = ComplexLeNet
gan_model_dict['Complex_ResNet-56'] = ComplexResNet
gan_model_dict['Complex_ResNet-110'] = ComplexResNet

# initialize our dataset dictionary
dataset_dict = {}
dataset_dict['CIFAR-10'] = load_cifar10_data
dataset_dict['CIFAR-100'] = load_cifar100_data

# initialize our U-net shape dictionary
unet_shape_dict = {}
unet_shape_dict['Complex_LeNet'] = (6,64,128,256,512)
unet_shape_dict['Complex_ResNet-56'] = (16,64,128,256,512)
unet_shape_dict['Complex_ResNet-110'] = (16,64,128,256,512)

# initialize our early stopping criteria
stop_criteria = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=3,
        verbose=False,
        mode='min'
    )

def train_model(args):
    """
    Function for training and testing a model.

    Inputs:
        args - Namespace object from the argument parser
    """

    # anomaly detection in case of changing models
    torch.autograd.set_detect_anomaly(False)

    # print the most important arguments given by the user
    print('----- MODEL SUMMARY -----')
    print('GAN Model: ' + args.gan_model)
    print('Dataset: ' + args.dataset)
    print('Epochs: ' + str(args.epochs))
    print('K value: ' + str(args.k))
    print('Learning rate: ' + str(args.lr))
    print('Batch size: ' + str(args.batch_size))
    print('Early stopping: ' + str(not args.no_early_stopping))
    print('Progress bar: ' + str(args.progress_bar))
    print('-------------------------')

    # make folder for the Lightning logs
    os.makedirs(args.log_dir, exist_ok=True)

    # load the data from the dataloader
    classes, trainloader, valloader, testloader = load_data_fn(
        args.dataset, args.batch_size, args.num_workers
    )

    # check whether to use early stopping
    if args.no_early_stopping:
        # initialize the Lightning trainer
        trainer = pl.Trainer(default_root_dir=args.log_dir,
                        gpus=1 if torch.cuda.is_available() else 0,
                        max_epochs=args.epochs,
                        progress_bar_refresh_rate=1 if args.progress_bar else 0)
    else:
        # initialize the Lightning trainer
        trainer = pl.Trainer(default_root_dir=args.log_dir,
                        gpus=1 if torch.cuda.is_available() else 0,
                        max_epochs=args.epochs,
                        progress_bar_refresh_rate=1 if args.progress_bar else 0,
                        callbacks=[stop_criteria])
    trainer.logger._default_hp_metric = None

    # seed for reproducability
    pl.seed_everything(args.seed)

    # initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gan_model = initialize_gan_model(args.gan_model, classes, args.lr, args.k)
    gan_model.load_state_dict(torch.load(args.load_gan))
    generator = gan_model.encoder.generator
    conv = gan_model.encoder.generator.encoding_layer
    enc_channels = unet_shapes(args.gan_model)
    model = UNet(generator=generator, enc_chs=enc_channels, lr=args.lr, encoding_layer=conv)

    # train the model
    print("Started training...")
    trainer.fit(model, trainloader, valloader)

    # save the model
    os.makedirs('./saved_models', exist_ok=True)
    torch.save(model.state_dict(), './saved_models/'+str(args.gan_model)+'Attacker_save')
    print('Training successfull')

    # test the model
    print("Started testing...")
    trainer.test(model=model, test_dataloaders=testloader)
    print('Testing successfull')

    # return the model
    return model

def initialize_gan_model(model='Complex_LeNet', num_classes=10, lr=3e-4, k=2):
    """
    Function for initializing a GAN model based on the given command line arguments.

    Inputs:
        model - String indicating the model to use. Default = 'Complex_LeNet'
        num_classes - Int indicating the number of classes. Default = 10
        lr - Float indicating the optimizer learning rate. Default = 3e-4
        k - Level of anonimity. k-1 fake features are generated
            to train the discriminator. Default = 2
    """

    # initialize the model if possible
    if model == "Complex_ResNet-110" or model == "ResNet-110":
        return gan_model_dict[model](num_classes, k, lr, num_blocks = [37,36,36])
    elif model == "Complex_ResNet-56" or model == "ResNet-56":
        return gan_model_dict[model](num_classes, k, lr, num_blocks = [19,18,18])
    elif model in gan_model_dict:
        return gan_model_dict[model](num_classes, k, lr)
    # alert the user if the given model does not exist
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model, str(gan_model_dict.keys()))

def load_data_fn(dataset='CIFAR-10', batch_size=256, num_workers=0):
    """
    Function for loading a dataset based on the given command line arguments.

    Inputs:
        dataset - String indicating the dataset to use. Default = 'CIFAR-10'
        batch_size - Int indicating the size of the mini batches. Default = 256
        num_workers - Int indicating the number of workers to use in the dataloader. Default = 0 (truly deterministic)
    """

    # load the dataset if possible
    if dataset in dataset_dict:
        return dataset_dict[dataset](batch_size, num_workers)
    # alert the user if the given dataset does not exist
    else:
        assert False, "Unknown dataset name \"%s\". Available datasets are: %s" % (dataset, str(dataset_dict.keys()))

def unet_shapes(model='Complex_LeNet'):
    """
    Function for selecting the correct unet dimensions for a given model.

    Inputs:
        model - String indicating the model to use. Default = 'Complex_LeNet'
    """

    # get the shape if possible
    if model in unet_shape_dict:
        return unet_shape_dict[model]
    # alert the user if the given model does not exist
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model, str(gan_model_dict.keys()))

if __name__ == '__main__':
    """
    Direct calling of the python file via command line.
    Handles the given hyperparameters.
    """

    # initialize the parser for the command line arguments
    parser = argparse.ArgumentParser()

    # model hyperparameters
    parser.add_argument('--gan_model', default='Complex_LeNet', type=str,
                        help='What model to use for the GAN. Default is Complex_LeNet.',
                        choices=['Complex_LeNet', 'Complex_ResNet-56', 'Complex_ResNet-110'])
    parser.add_argument('--dataset', default='CIFAR-10', type=str,
                        help='What dataset to use. Default is CIFAR-10.',
                        choices=['CIFAR-10', 'CIFAR-100', 'CUB-200'])

    # dataloader hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size. Default is 64.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. Default is not 0 (truly deterministic).')

    # training hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs. Default is 10.')
    parser.add_argument('--k', default=2, type=int,
                        help='Level of anonimity to use during training. k-1 fake features are generated to train the encoder. Default is 2,')
    parser.add_argument('--log_dir', default='attacker_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs are created. Default is attacker_logs/.')
    parser.add_argument('--load_gan', default=None, type=str, required=True,
                        help='Directory where the model for the GAN is stored. Is required.')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator. Disabled by default.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results. Default is 42.')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Disable early stopping. Enabled by default.')

    # optimizer hyperparameters
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='Learning rate to use. Default is 3e-4.')

    # parse the arguments
    args = parser.parse_args()

    # train the model with the given arguments
    train_model(args)
