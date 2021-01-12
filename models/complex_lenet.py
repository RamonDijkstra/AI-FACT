# Adapted from https://github.com/wavefrontshaping/complexPyTorch
# from complexPyTorch-master.complexFunctions import *
# from complexPyTorch-master.complexLayers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np


class EncoderGenerator(nn.Module):

    def __init__(self, k):
        """
        """
        super().__init__()

        # k - 1 features will be generated
        self.k = k

    def forward(self, a):
        """
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # We use the feature to generate k - 1 fake features
        a_size = a.shape
        magnitude = torch.norm(a)

        # b is normalized to have same magnitude as a

        b = torch.randn(a_size, device=device)
        divider = torch.norm(b) / magnitude
        b = b / divider

        # Tensor of angles that rotate real feature, sampled from uniform
        # distribution between 0 and pi (excluding 0 itself)
        thetas = torch.cuda.FloatTensor(self.k-1).uniform_(0, np.pi)


        ###Set theta and b for real feature, such that decoder can use it later
        #theta ===
        #b ===



        # Do we also add the real feature (i.e. a) to x ..?
        x = torch.empty(self.k, a_size[0], a_size[1], a_size[2], a_size[3], dtype=torch.cfloat, device=device)
        a = a.cpu()
        b = b.cpu()
        thetas = thetas.cpu()
        for i in range(0, self.k-1):
            x[i] = (a + b *1j) * torch.exp(1j * thetas[i])

        a = a.to(device)
        b = b.to(device)
        thetas = thetas.to(device)
        return x, thetas

class EncoderDiscriminator(nn.Module):

    def __init__(self):
        """
        """
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, x):
        x = x.real
        a = a.reshape(1, a.shape[0], a.shape[1], a.shape[2], a.shape[3])
        cat = torch.cat((a,x))
      
        return self.sigmoid(cat)

class LenetEncoder(nn.Module):
    def __init__(self, k):

        super(LenetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        ###Moet anders: Generator en discriminator optimaliseren op GAN manier, om zo de Conv weights te leren, daarna batch door conv halen
        ### roteren en complexiseren
        self.generator = EncoderGenerator(k)
        self.discriminator = EncoderDiscriminator()

    def forward(self, x):
        # conv1 is the encoder which maps the input to the feature 
        a = self.conv1(x)
        generated, theta = self.generator(a)
        out = self.discriminator(a, generated)
        #Roteer a en complexiseer
        return a,theta,out

class LenetProcessingModule(nn.Module):
    def __init__(self):
        super(LenetProcessingModule, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def complex_relu(x):
	    #Zou dit werken? Sowieso
	    c = torch.ones(x.shape, device=self.device)
	    return torch.linalg.norm(x, float('inf')) / torch.max(torch.linalg.norm(x, float('inf')), c) * x
    
    def forward(self, x):
        #print(x.shape)
        x = complex_relu(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)

        x = self.conv2(x)
        x = complex_relu(x)
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)

        x = self.fc1(x)
        x = complex_relu(x)

        x = self.fc2(x)
        x = complex_relu(x)

        x = self.fc3(x)
        
        return x

class LenetDecoder(nn.Module):
    def __init__(self):
        super(LenetDecoder, self).__init__()

        self.softmax = nn.Softmax()

    def forward(self, x, theta):

    	#Eerst terug roteren, dan .real dan softmax
    	x = x * torch.exp(-1j * theta)
    	x = x.real
    	
        x = self.softmax(x)
       	return x

class ComplexLenet(nn.Module):
    def __init__(self, k=5, device):
        super(ComplexLenet, self).__init__()

        self.device = device

        self.encoder = LenetEncoder(k)
        self.proccessing_module = LenetProcessingModule()
        self.decoder = LenetDecoder()

    def forward(self, x):
        #x is an image batch
        x, theta, discriminator_logits = self.encoder(x)
        x = self.proccessing_module(x)
        x = self.decoder(x, theta)

        return x, discriminator_logits

