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
Helper function that loads CIFAR-10 data
"""

# pytorch imports
import torch
import torchvision
import torchvision.transforms as transforms

def load_data(batch_size=128, num_workers=0):
    '''
    Function loads the CIFAR-10 dataset and splits into
    train, validation and test sets.

    Inputs:
        batch_size - Int indicating the batch size. Default = 256
        num_workers - Int indicating the number of workers for loading
            the data. Default = 0 (truly deterministic)
    '''

    # normalize the input
    transform = transforms.Compose(
    [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])

    # load the training and test dataset
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                        download=True, transform=transform)

    # split the test dataset into test and validation
    val_split = int(len(testset)/2)
    test_split = len(testset) - val_split
    valset, testset = torch.utils.data.random_split(testset, [val_split, test_split])

    # create the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)

    # 10 classes for CIFAR-10
    num_classes = 10

    # return the classes and dataloaders
    return num_classes, trainloader, valloader, testloader
