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
Main training file
"""

# basic imports
import time
from os import listdir
from os.path import isfile, join

# pytorch imports
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# import models
from models.lenet.lenet import *
from models.lenet.complex_lenet import *
from models.alexnet.alexnet import *
from models.alexnet.complex_alexnet import *
from models.resnet56.resnet56 import *
from models.resnet56.complex_resnet56 import *
from models.resnet110.complex_resnet110 import *
from models.vgg16.vgg16 import *
from models.vgg16.vgg16_complex import *

# import dataloaders
from dataloaders.cifar10_loader import load_data as load_cifar10_data
from dataloaders.cifar100_loader import load_data as load_cifar100_data
from dataloaders.celeba_loader import load_data as load_celeba_data
from dataloaders.cub2011_loader import load_data as load_cub200_data

# initialize our model dictionary
model_dict = {}
model_dict['LeNet'] = LeNet
model_dict['Complex_LeNet'] = ComplexLeNet
model_dict['AlexNet'] = AlexNet
model_dict['Complex_AlexNet'] = ComplexAlexNet
model_dict['ResNet-56'] = ResNet
model_dict['ResNet-110'] = ResNet
model_dict['Complex_ResNet-56'] = ComplexResNet56
model_dict['Complex_ResNet-110'] = ComplexResNet110
model_dict['VGG-16'] = VGG16
model_dict['Complex_VGG-16'] = Complex_VGG16

### Note that Complex ResNet are the alpha variants

# initialize our dataset dictionary
dataset_dict = {}
dataset_dict['CIFAR-10'] = load_cifar10_data
dataset_dict['CIFAR-100'] = load_cifar100_data
dataset_dict['CelebA'] = load_celeba_data
dataset_dict['CUB-200'] = load_cub200_data

# initialize our early stopping criteria
stop_criteria = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=10,
        verbose=False,
        mode='max'
    )

def train_model(args):
    """
    Function for training and testing a model.

    Inputs:
        args - Namespace object from the argument parser
    """

    # DEBUG
    torch.autograd.set_detect_anomaly(True)

    # print the most important arguments given by the user
    print('----- MODEL SUMMARY -----')
    print('Model: ' + args.model)
    print('Dataset: ' + args.dataset)
    print('Epochs: ' + str(args.epochs))
    print('K value: ' + str(args.k))
    print('Learning rate: ' + str(args.lr))
    print('Early stopping: ' + str(not args.no_early_stopping))
    print('-------------------------')

    # keep track of experiment elapsed time
    experiment_start = time.time()

    # make folder for the Lightning logs
    os.makedirs(args.log_dir, exist_ok=True)

    # load the data from the dataloader
    num_classes, trainloader, valloader, testloader = load_data(
        args.dataset, args.batch_size, args.num_workers
    )

    # check whether to use early stopping
    if args.no_early_stopping:
        # initialize the Lightning trainer
        trainer = pl.Trainer(default_root_dir=args.log_dir,
                        checkpoint_callback=ModelCheckpoint(
                            save_weights_only=True),
                        gpus=1 if torch.cuda.is_available() else 0,
                        max_epochs=args.epochs,
                        progress_bar_refresh_rate=1 if args.progress_bar else 0)
    else:
        # initialize the Lightning trainer
        trainer = pl.Trainer(default_root_dir=args.log_dir,
                        checkpoint_callback=ModelCheckpoint(
                            save_weights_only=True),
                        gpus=1 if torch.cuda.is_available() else 0,
                        max_epochs=args.epochs,
                        progress_bar_refresh_rate=1 if args.progress_bar else 0,
                        callbacks=[stop_criteria])
    trainer.logger._default_hp_metric = None

    # seed for reproducability
    pl.seed_everything(args.seed)

    # initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(args.model, num_classes, args.lr, args.k)

    # load the pre-trained model if directory has been given
    if args.load_dict:
        print('Loading model..')
        model_dir = args.load_dict
        checkpoint_dir = model_dir + "\checkpoints\\"
        last_ckpt = [f for f in listdir(checkpoint_dir) if isfile(join(checkpoint_dir, f))][-1:][0]
        checkpoint_path = checkpoint_dir + last_ckpt
        hparams_file = model_dir + "\hparams.yml"
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams_file=hparams_file
        )
        print('Model successfully loaded')
    else:
        print('Training model..')
        # train the model
        trainer.fit(model, trainloader, valloader)
        print('Training successfull')

    # show the progress bar if enabled
    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # keep track of testing elapsed time
    test_start = time.time()

    # test the model
    print('Testing model..')
    trainer.test(model=model, test_dataloaders=testloader)
    print('Testing successfull')

    # print the elapsed time of experiment and testing
    end = time.time()
    print('Testing run-time: ' + format_seconds_to_hhmmss(end - test_start))
    print('Experiment run-time: ' + format_seconds_to_hhmmss(end - experiment_start))

    # return the model
    return model

def initialize_model(model='Complex_LeNet', num_classes=10, lr=3e-4, k=2):
    """
    Function for initializing a model based on the given command line arguments.

    Inputs:
        model - String indicating the model to use. Default = 'Complex_LeNet'
        num_classes - Int indicating the number of classes. Default = 10
        lr - Float indicating the optimizer learning rate. Default = 3e-4
        k - Level of anonimity. k-1 fake features are generated
            to train the discriminator. Default = 2
    """

    # initialize the model if possible
    if model in model_dict:
        if model == 'ResNet-110':
            num_blocks = [37,36,36]
            return model_dict[model](num_classes, k, lr, num_blocks)
        return model_dict[model](num_classes, k, lr)
    # alert the user if the given model does not exist
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model, str(model_dict.keys()))

def load_data(dataset='CIFAR-10', batch_size=256, num_workers=0):
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

def format_seconds_to_hhmmss(seconds):
    """
    Function for converting seconds to a string.

    Inputs:
        seconds - Float indicating the number of seconds elapsed.
    Outputs:
        string - Formatted string of the elapsed time.
    """

    try:
        hours = seconds // (60*60)
        seconds %= (60*60)
        minutes = seconds // 60
        seconds %= 60
        return "%02i:%02i:%02i" % (hours, minutes, seconds)
    except:
        return 'Invalid time'

if __name__ == '__main__':
    """
    Direct calling of the python file via command line.
    Handles the given hyperparameters.
    """

    # initialize the parser for the command line arguments
    parser = argparse.ArgumentParser()

    # model hyperparameters
    parser.add_argument('--model', default='Complex_LeNet', type=str,
                        help='What model to use. Default is Complex_LeNet.',
                        choices=['LeNet', 'Complex_LeNet', 'AlexNet', 'Complex_AlexNet', 'ResNet-56', 'Complex_ResNet-56', 'ResNet-110', 'Complex_ResNet-110', 'VGG-16', 'Complex_VGG-16'])
    parser.add_argument('--dataset', default='CIFAR-10', type=str,
                        help='What dataset to use. Default is CIFAR-10.',
                        choices=['CIFAR-10', 'CIFAR-100', 'CelebA', 'CUB-200'])

    # dataloader hyperparameters
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Minibatch size. Default is 4.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. Default is not 0 (truly deterministic).')

    # training hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs. Default is 10.')
    parser.add_argument('--k', default=2, type=int,
                        help='Level of anonimity to use during training. k-1 fake features are generated to train the encoder. Default is 2,')
    parser.add_argument('--log_dir', default='complex_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs are created. Default is GAN_logs/.')
    parser.add_argument('--load_dict', default=None, type=str,
                        help='Directory where the model is stored. Default is inference_attack_model.')
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
