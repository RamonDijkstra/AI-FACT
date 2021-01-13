# Adapted from https://github.com/wavefrontshaping/complexPyTorch
# from complexPyTorch-master.complexFunctions import *
# from complexPyTorch-master.complexLayers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np


class EncoderGenerator(nn.Module):

    def __init__(self, k, device):
        """
        """
        super().__init__()

        # k - 1 features will be generated
        self.k = k
        self.device = device

    def forward(self, a):
        """
        """

        # We use the feature to generate k - 1 fake features

        a_size = a.shape # features after convolution

        magnitude = torch.norm(a).item()


        # = variable for K
        vec = torch.normal(0, 1, size=tuple((a_size[0]*self.k,a_size[1],a_size[2],a_size[3])))
        mag = torch.sqrt(torch.sum(torch.square(vec)).type(torch.FloatTensor))
        res = vec/mag
        b = res*magnitude
        b = b.to(self.device)


        thetas = torch.Tensor(self.k*a_size[0]).uniform_(0, np.pi).to(self.device)
 
        ###Set theta for real feature, such that decoder can use it later
       
        #Only works for K = 2 ----- Make variable perhaps
        a = torch.cat([a,a],dim=0)
      
        x = torch.empty(self.k*a_size[0], a_size[1], a_size[2], a_size[3], device=self.device)

        thetas = thetas.cpu()
        thetas = (1j * thetas).exp()
        thetas = thetas.to(self.device)


        thetas = thetas.reshape(self.k*a_size[0],1,1,1)

        x = (a + b *1j) * thetas
        return x, thetas[:a_size[0]].real.squeeze()

class EncoderDiscriminator(nn.Module):

	def __init__(self):
		"""
		"""
		super().__init__()
		self.linear = nn.Linear(6*28*28,1)
		self.sigmoid = nn.Sigmoid()

	def forward(self,generated, a):
		a = a.view(a.shape[0],6*28*28)
		generated= generated[a.shape[0]:,:,:]
		generated=generated.real
		generated = generated.view(generated.shape[0],6*28*28)
		x = torch.cat((a,generated),dim=0)
		x = self.linear(x)
		return self.sigmoid(x).squeeze()

class LenetEncoder(nn.Module):
    def __init__(self, k, device):

        super(LenetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        ###Moet anders: Generator en discriminator optimaliseren op GAN manier, om zo de Conv weights te leren, daarna batch door conv halen
        ### roteren en complexiseren
        self.generator = EncoderGenerator(k, device)
        self.discriminator = EncoderDiscriminator()

    def forward(self, x):
        # conv1 is the encoder which maps the input to the feature 
        a = self.conv1(x)
        generated, theta = self.generator(a)
        #Make for variable K
        labels = torch.cat([torch.ones(a.shape[0]),torch.zeros(a.shape[0])], dim=0)

        #print("Lables",labels)


        out = self.discriminator(generated,a)
        return generated[:a.shape[0],:,:,:] ,theta,out, labels

class LenetProcessingModule(nn.Module):
    def __init__(self, device):
        super(LenetProcessingModule, self).__init__()

        self.device = device
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

   
    
    def forward(self, x):
        #print(x.shape)
        x = complex_relu(x,self.device)
        #print(x.shape)
        indices = complex_max_pool(x,self.pool)
        #print(x.shape)
        x = x[indices]
        x = self.conv2(x)
        x = complex_relu(x,self.device)
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)

        x = self.fc1(x)
        x = complex_relu(x,self.device)

        x = self.fc2(x)
        x = complex_relu(x,self.device)

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
    def __init__(self, device, k=5 ):
        super(ComplexLenet, self).__init__()

        self.device = device

        self.encoder = LenetEncoder(k, self.device)
        self.proccessing_module = LenetProcessingModule(self.device)
        self.decoder = LenetDecoder()

    def forward(self, x):
        #x is an image batch
        x, theta, discriminator_logits, labels = self.encoder(x)
        #x = self.proccessing_module(x)
        #x = self.decoder(x, theta)

        return x, discriminator_logits, labels

