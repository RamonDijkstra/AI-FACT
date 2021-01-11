# Adapted from https://github.com/wavefrontshaping/complexPyTorch
from complexPyTorch-master.complexFunctions import *
from complexPyTorch-master.complexLayers import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class EncoderGenerator(nn.Module):

    def __init__(self, k, size, magnitude):
        """
        """
        super().__init__()

        # k - 1 features will be generated
        self.k = k

        # b is normalized to have same magnitude as a
        b = torch.randn(size)
        divider = torch.linalg.norm(b) / magnitude
        self.b = b / divider

        # Tensor of angles that rotate real feature, sampled from uniform
        # distribution between 0 and pi (excluding 0 itself)
        self.thetas = nn.Parameter(torch.empty(k-1))
        self.thetas = torch.cuda.FloatTensor(k-1).uniform_(0, np.pi)


    def forward(self, a):
        """
        """

        # Do we also add the real feature (i.e. a) to x ..?
        x = torch.empty(self.k)
        x[0] = a
        for i in range(1, self.k):
            x[i] = (a + self.b *1j) * torch.exp(1j * self.thetas[i])
        return x

class EncoderDiscriminator(nn.Module):

    def __init__(self):
        """
        """
        super().__init__()
        self.sigmoid = torch.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x.real)


class LenetProcessingModule(nn.Module):
    def __init__(self, k):
        super(LenetProcessingModule, self).__init__()

        self.conv2 = ComplexConv2d(6, 16, 3, bias=False)
        # an affine operation: y = Wx + b
        self.fc1 = ComplexLinear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = ComplexLinear(120, 84)
        self.fc3 = ComplexLinear(84, 10)
    
    def forward(self, x):
        temp = self.conv2(x)
        temp = self.fc1(temp)
        temp = self.fc2(temp)
        out = self.fc3(temp)
        
        return out

class LenetDecoder(nn.Module):
    def __init__(self):
        super(LenetDecoder, self).__init__()

    def forward(self, x):
        pass


class Discriminator(nn.Module):

    def __init__():
        """
        """
        super().__init__()


    def forward(self, x):
        """
        """
        preds = self.layers(x)

        return preds

class LenetEncoder(nn.Module):
    def __init__(self, k=4):

        super(LenetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.k = k
        self.generator = Generator()

    def forward(self, x):
        # conv1 is the encoder which maps the input to the feature 
        a = self.conv1(x)
        # We use the feature to generate k - 1 fake features
        size = a.shape
        magnitude = torch.linalg.norm(a)
        k = self.generator(self.k, size, magnitude)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
