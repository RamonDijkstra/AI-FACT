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
Helper function that loads CUB-200 2011 data
"""

# basic imports
import os
import pandas as pd
import tarfile
from os import path

# pytorch imports
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def load_data(batch_size=128, num_workers=0):
    '''
    Function loads the CUB-200 dataset and splits into
    train, validation and test sets.

    Inputs:
        batch_size - Int indicating the batch size. Default = 256
        num_workers - Int indicating the number of workers for loading
            the data. Default = 0 (truly deterministic)
    '''

    # normalize the input
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize(size=(56,56))]
    )

    # create path with images
    if not path.exists('./data/CUB_200_2011/CUB_200_2011/images'):
        # check if tar file has been downloaded
        if not path.exists('./data/CUB_200_2011/CUB_200_2011.tgz'):
            # alert user of missing tar file
            assert False, "CUB-200 tar file missing. Place inside './data/CUB200_2011' folder. Please download via https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view"
        print('Unpacking tar file..')
        with tarfile.open(os.path.join('./data/CUB_200_2011', 'CUB_200_2011.tgz'), "r:gz") as tar:
            tar.extractall(path='./data/CUB_200_2011')
        print('Unpacking finished')

    # retrieve the dataset
    data_set = ImageFolder('./data/CUB_200_2011/CUB_200_2011/images', transform=transform)

    # split the data in 70% train, 15% test and 15% validation
    train_split = int(0.7 * len(data_set))
    val_test_split = len(data_set) - train_split
    trainset, val_testset = torch.utils.data.random_split(
        data_set, [train_split, val_test_split]
    )
    val_split = int(len(val_testset)/2)
    test_split = len(val_testset) - val_split
    valset, testset = torch.utils.data.random_split(
        val_testset, [val_split, test_split]
    )

    # create the dataloaders
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 200 classes for CUB-200
    num_classes = 200

    # return the classes and dataloaders
    return num_classes, trainloader, valloader, testloader
