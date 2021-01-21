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
Model for inference attacks
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


class InferenceAttacker(nn.Module):
	"""
	Inference attack model
	"""

    def __init__(self, dataset='CIFAR-100', input_shape=[1, 28, 28], output_shape=[1, 28, 28]):
        """
        Network for inference attacks based on the ResNet-50 or ResNet-56.

        Inputs:
            dataset - Defines whether to use ResNet-50 or ResNet-56.
                Resnet-50 for 'CelebA'
                Resnet-56 for 'CIFAR-100'
			input_shape - Shape of the input images.
            output_shape - Shape of the output inferred attributes.
        """
        super().__init__()
        
        # create a normalization layer for the input so it can be accepted by the ResNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        # create either ResNet-50 or ResNet-56
        if (dataset == 'CIFAR-100'):
            self.network = models.resnet50()
        elif (dataset == 'CelebA'):
            self.network = models.resnet56()
		
        raise NotImplementedError

    def forward(self, encoded_feature):
        """
        Inputs:
            encoded_feature - Input batch of encoded features. Shape: [B, ?, ?]
        Outputs:
			reconstructed_image - Generated original image of shape 
				[B,image_shape[0],image_shape[1],image_shape[2]]
        """
		
        raise NotImplementedError

    def training_step(self):
        # raw images to identify hidden properties
        # CIFAR-100 - ResNet-56, as prototype model.
        # attacker net also ResNet-56,

    def test_step(self):
        #Use outcome of inversion attack 1 to identify hidden properties and compare

        # INPUT: dec(a*) from inversion attack 1
        # According to the paper:
        #   accuracy of the DNN on CIFAR-100 was evaluated by the classification error of the major 20 superclasses, 
        #   and the privacy was gauged by the classification error of the 100 minor classes.




    @property
    def device(self):
        """
        Property function to get the device on which the model is
        """
        return next(self.parameters()).device