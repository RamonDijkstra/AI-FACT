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
Helper function that loads CIFAR10 data
"""

import torch
import torchvision
import torchvision.transforms as transforms

def load_data(batch_size, num_workers):
	'''
    loads the CIFAR10 dataset and splits into
	train and test sets
    '''
	
	transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

	testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

	classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return classes, trainloader, testloader