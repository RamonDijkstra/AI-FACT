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

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# import models
from models.lenet.lenet import *
from models.lenet.complex_lenet import *
from models.alexnet.alexnet import *
from models.alexnet.complex_alexnet import *
from models.resnet56.resnet56 import *
from models.resnet56.complex_resnet56 import *
from models.resnet110.resnet110 import *
from models.resnet110.complex_resnet110 import *


# import dataloaders
from dataloaders.cifar10_loader import load_data as load_cifar10_data
from dataloaders.cifar100_loader import load_data as load_cifar100_data
from dataloaders.celeba_loader import load_data as load_celeba_data

# initialize our model dictionary
model_dict = {}
model_dict['LeNet'] = LeNet
model_dict['Complex_LeNet'] = ComplexLeNet
model_dict['AlexNet'] = AlexNet
model_dict['Complex_AlexNet'] = ComplexAlexNet
model_dict['ResNet-110'] = ResNet110
model_dict['ResNet-56'] = ResNet56
model_dict['Complex_ResNet-56'] = ComplexResNet56
model_dict['Complex_ResNet-110'] = ComplexResNet110

### Note that Complex ResNet are the alpha variants

# initialize our dataset dictionary
dataset_dict = {}
dataset_dict['CIFAR-10'] = load_cifar10_data
dataset_dict['CIFAR-100'] = load_cifar100_data
dataset_dict['CelebA'] = load_celeba_data

def train_model(args):
    """
    Function for training and testing a model.
    
    Inputs:
        args - Namespace object from the argument parser
    """
    
    # DEBUG
    torch.autograd.set_detect_anomaly(True)
    
    # make folder for the Lightning logs
    os.makedirs(args.log_dir, exist_ok=True)
    
    # load the data from the dataloader  
    print(args.dataset)
    classes, trainloader, testloader = load_data(args.dataset, args.batch_size, args.num_workers)


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
    num_classes = classes

    # Depending on model
    #model = ComplexResnet(num_classes=num_classes, lr=args.lr, k=args.k, num_blocks=[19,18,18])
    # model = ComplexLenet(num_classes=num_classes, lr=args.lr, k=args.k)
    model = initialize_model(args.model, num_classes, args.lr, args.k)

    print(model)

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

def initialize_model(model='LeNet', num_classes=10, lr=3e-4, k=2):
    """
    Function for initializing a model based on the given command line arguments.
    
    Inputs:
        model - String indicating the model to use. Default = 'LeNet'
        num_classes - Int indicating the number of classes. Default = 10
        lr - Float indicating the optimizer learning rate. Default = 3e-4
        k - Level of anonimity. k-1 fake features are generated
            to train the discriminator. Default = 2
    """
    
    # initialize the model if possible
    if model in model_dict:
        return model_dict[model](num_classes, k, lr)
    # alert the user if the given model does not exist
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model_name, str(model_dict.keys()))
        
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
                        choices=['LeNet', 'Complex_LeNet', 'AlexNet', 'Complex_AlexNet', 'ResNet-56', 'Complex_ResNet-56', 'ResNet-110', 'Complex_ResNet-110'])
    parser.add_argument('--dataset', default='CIFAR-10', type=str,
                        help='What dataset to use. Default is CIFAR-10.',
                        choices=['CIFAR-10', 'CIFAR-100', 'CelebA'])
    
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
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator. Disabled by default.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results. Default is 42.')
                        
    # optimizer hyperparameters
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='Learning rate to use. Default is 3e-4.')
    
    # parse the arguments 
    args = parser.parse_args()

    # train the model with the given arguments
    train_model(args)
