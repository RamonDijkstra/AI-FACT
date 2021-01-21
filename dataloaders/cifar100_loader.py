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
Helper function that loads CIFAR100 data
"""

import torch
import torchvision
import torchvision.transforms as transforms

def load_data(batch_size=128, num_workers=2):
    '''
    loads the CIFAR100 dataset and splits into
    train and test sets
    '''
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
    trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True,
                                        download=True, transform=transform)
                                        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
                                          
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False,
                                       download=True, transform=transform)
                                       
    val_split = int(len(testset)/2)
    test_split = len(testset) - val_split
    valset, testset = torch.utils.data.random_split(testset, [val_split, test_split])

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
                                                
    num_classes = 100

    return num_classes, trainloader, valloader, testloader