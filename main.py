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
Main python file
"""

#SHOUTOUT NAAR PHILLIPE GEVEN

import argparse
import os

from models.lenet import *
from dataloaders.cifar10_loader import load_data
from models.complex_lenet_v2 import *

import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train(args):
    """
    Function for training and testing a the complex encryption model.
    Inputs:
        args - Namespace object from the argument parser
    """
    
    # initialize the device to train the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the data from the dataloader
    classes, trainloader, testloader = load_data(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # initialize the network
    net = ComplexLenet(device, num_classes=10)
    net = net.to(device)
    
    # initialize the different loss criteria
    # TODO: change net_criterion to crossentropyloss if you use default lenet
    net_criterion = nn.NLLLoss()
    # TODO: check if gan_criterion is indeed BCEWithLogitsLoss
    gan_criterion = nn.BCEWithLogitsLoss()


    # initialize the different optimizers
    # TODO: misschien Adam? Optimizer voor het hele model (min g,Phi,d)
    model_optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    # TODO: checken if inderdaad SGD? Optimizer om de discriminator leren (max D)
    gan_optimizer = optim.SGD(net.encoder.discriminator.parameters(), lr=args.lr, momentum=0.9)


    # start loop for the given number of epochs
    for epoch in range(args.epochs):
        # keep track of the model and gan loss
        model_loss_value = 0.0
        gan_loss_value = 0.0
        
        # DEBUG
        last_labels = None
        last_predictions = None
        torch.autograd.set_detect_anomaly(True)
        
        # start loop over the different batches in the trainloader
        for i, data in enumerate(trainloader, 0):
            # get the images and labels from the data
            images, labels = data[0].to(device), data[1].to(device)
            
            # run the images through the network
            outputs, discriminator_predictions, discriminator_labels = net(images)
            discriminator_predictions, discriminator_labels = discriminator_predictions.to(device), discriminator_labels.to(device)
            gan_optimizer.zero_grad()
            
            # calculate the loss of the GAN
            gan_loss = gan_criterion(discriminator_predictions, discriminator_labels)
            gan_loss.backward()
            
            # make the GAN perform better by setting a step with the optimizer
            gan_optimizer.step()
            
            # DEBUG
            last_labels = discriminator_labels
            last_predictions = discriminator_predictions

            #print(discrim_loss.item())
            
            model_optimizer.zero_grad()
            print("iuashfuisdhfisodsdfhi", outputs.shape)
            print(labels.shape)
            task_loss = net_criterion(outputs, labels)
            task_loss.backward()

            model_optimizer.step()


            #model_optimizer.zero_grad()
            #task_loss = net_criterion(outputs, labels)
            #task_loss.backward()
            #model_optimizer.step()

            ### Nadenken over volgorde Loss en optimizen

            # zero the parameter gradients
            # Loss discriminator.backward
            # Step optim discrim (max)
            # 
            # zero model_optim gradients
           	# task loss
           	# overall loss = task loss -/+ loss discriminator
           	# overall loss.backward
           	#step model_opti

            # forward + backward + optimize
            #loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()

            # print statistics
            #discrim_running_loss += discrim_loss.item()
            #print(discrim_loss.item())
            #running_loss += loss.item()
            #if i % 2000 == 1999:    # print every 2000 mini-batches
            	#print('[%d, %5d] loss: %.3f' %
            	#(epoch + 1, i + 1, running_loss / 2000))\
                
            #discriminator_loss_value = discrim_loss.item()
            #task_loss_value = task_loss.item()
        print('epoch {} disc loss: {}'.format(epoch + 1, discriminator_loss_value))
        #print('epoch {} task loss: {}'.format(epoch + 1, task_loss_value))
            	
    print('Finished Training')
    
    print('Final labels')
    print(last_labels)
    print('Final predictions')
    print(last_predictions)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    #CHECK HYPERPARAMETERS

    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='MLP', type=str,
                        help='What model to use in the VAE',
                        choices=['MLP', 'CNN'])
    parser.add_argument('--z_dim', default=20, type=int,
                        help='Dimensionality of latent space')
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
    parser.add_argument('--epochs', default=20, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='VAE_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train(args)
