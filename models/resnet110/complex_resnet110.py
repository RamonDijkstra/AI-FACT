####0 Almost all rights reserved to Philipp Lippe
### https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html


## Standard libraries
import os
import numpy as np 
import random
from PIL import Image
from types import SimpleNamespace

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

#Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import *
from models.encoder.GAN import EncoderGAN

class ComplexResNet110(pl.LightningModule):
    """
	Complex LeNet model
	"""

    def __init__(self, num_classes=10, k=2, lr=1e-3, num_blocks=[37,36,36]):
        """
        Complex LeNet network

        Inputs:
            num_classes - Number of classes of images, Default = 10
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. Default = 2
            lr - Learning rate to use for the optimizer. Default = 1e-3
        """
        super(ComplexResNet110, self).__init__()       
        self.save_hyperparameters()

        # save the inputs
        self.num_classes = num_classes
        self.lr = lr
        self.k = k


        #Variable to make sure the linear layer in the discriminator in the GAN has the right shape
        Discriminator_linear_shape = 16*16*16

        # Creating the ResNet blocks
        blocks = []
        blocks.append(nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False))
        blocks.append(nn.BatchNorm2d(16))
        blocks.append(nn.ReLU())
        for i in range(num_blocks[0]):
            subsample = (i == 0)
            blocks.append(
                    ResNetBlock(c_in=16,
                                             act_fn=nn.ReLU(),
                                             subsample=subsample,
                                             c_out=16)
                )

        input_net = nn.Sequential(*blocks)


        # initialize the different modules of the network
        self.encoder_layers = input_net
        self.encoder = EncoderGAN(input_net,Discriminator_linear_shape, self.k, self.lr)

        #self.encoder = GAN(self.k, self.lr, num_blocks[0])
        self.proccessing_module = Resnet_Processing_module(num_blocks[1])
        self.decoder = Resnet_Decoder(num_blocks[2], self.num_classes)
        self.softmax = nn.Softmax()

        print("CLASSES", self.num_classes)
        
        # initialize the loss function of the complex LeNet
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """
        Function to configure the optimizers
        """
        
        # initialize optimizer for the entire model
        model_optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # return the optimizer
        return model_optimizer

    def training_step(self, batch, optimizer_idx):
        """
        Inputs:
            image_batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            training - Boolean value. Default = True
                True when training
                False when using in application
        Outputs:
			decoded_feature - Output batch of decoded real features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
            discriminator_predictions - Predictions from the discriminator. Shape: [B * k, 1]
            labels - Real labels of the encoded features. Shape: [B * k, 1]
        """
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, optimizer_idx)

        # send the encoded feature to the processing unit
        #print(out.shape)
        #print(out[0])


        out = self.proccessing_module(out)
        
        # # decode the feature from
        out = self.decoder(out, thetas)

        #print(result.shape)

        # Log the train accuracy
        result = self.softmax(out)
        preds = out.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        model_loss = self.loss_fn(out, labels)
        
        loss = gan_loss + model_loss

        # log the loss
        self.log("generator/loss", gan_loss)
        self.log("model/loss", model_loss)
        self.log("total/loss", loss)
        self.log('train_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards

        return loss

    def validation_step(self, batch, optimizer_idx):
        """
        Validation step of the standard ResNet-56 model.
        
        Inputs:
            batch - Input batch of images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
        Outputs:
            loss - Tensor representing the model loss.
        """
        
        # divide the batch in images and labels
        x, labels = batch
        
        # run the image batch through the network
        gan_loss, out, thetas = self.encoder(x, False)
        
        out = self.proccessing_module(out)

        out = self.decoder(out, thetas)

        # log the validation accuracy
        preds = self.softmax(out)
        preds = preds.argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('val_acc', acc)
        
        # log the validation loss
        loss = self.loss_fn(out, labels)
        self.log("val_loss", loss)

        # return the loss
        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, False)
        
        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from
        out = self.decoder(out, thetas)
        result = self.softmax(out)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        loss = self.loss_fn(out, labels)

        self.log("test_loss", loss)
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards


class Resnet_Processing_module(nn.Module):
    def __init__(self,  num_blocks):

        super().__init__()
        self.num_blocks = num_blocks

        blocks = []     # Creating the ResNet blocks -- Normal for now, need to make complex layers
        for i in range(self.num_blocks):
            subsample = (i == 0)
            #print(subsample)
            blocks.append(
            ResNetBlock(c_in=32 if not subsample else 16,
                            act_fn=nn.ReLU(),
                            subsample=subsample,
                            c_out=32,
                            complex=True)
                    )
            self.blocks = nn.Sequential(*blocks)

    def forward(self, encoded_batch):
        out = self.blocks(encoded_batch)
        return out

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device




class Resnet_Decoder(nn.Module):
    def __init__(self, num_blocks, num_classes):

        super().__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        blocks = []
        for i in range(self.num_blocks):
            subsample = (i == 0)

            blocks.append(
                    ResNetBlock(c_in=64 if not subsample else 32,
                                    act_fn=nn.ReLU(),
                                    subsample=subsample,
                                    c_out=64)
                    )
        
        self.blocks = nn.Sequential(*blocks)
        
        self.output_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(64, self.num_classes)
            )

    def forward(self,encoded_batch, thetas):
        #Rerotate
        #print(encoded_batch.shape)
        #print(thetas.shape)
        print(encoded_batch.shape)
        print(thetas.shape)
        decoded_batch = encoded_batch * torch.exp(-1j * thetas.squeeze())[:,None,None,None]



                #Make real

        decoded_batch = decoded_batch.real
        #print(decoded_batch.shape)

        # Go through Blocks and output net
        decoded_batch = self.blocks(decoded_batch)

        out = self.output_net(decoded_batch)
        


        
        return out


class ResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn=nn.ReLU(), subsample=False, c_out=-1, complex=False):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        self.complex = complex

        if not subsample:
            c_out = c_in
        #print(subsample)
        #print(c_in,c_out)
        if self.complex:
            self.Conv1_real =   nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False) # No bias needed as the Batch Norm handles it
            self.Conv1_imag =   nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False)
            #self.Batch_norm = nn.BatchNorm2d(c_out),
            #act_fn,
            self.Conv2_real = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
            self.Conv2_imag = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
            #nn.BatchNorm2d(c_out)
            
        
            # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
            self.downsample_conv_real = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False) if subsample else None
            self.downsample_conv_imag = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False) if subsample else None
            #self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
            #self.act_fn = act_fn
            

            #self.act_fn = act_fn

        else:
            # Network representing F
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False), # No bias needed as the Batch Norm handles it
                nn.BatchNorm2d(c_out),
                act_fn,
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            
            # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
            self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
            #self.act_fn = act_fn
            self.act_fn = act_fn

        
    def forward(self, x):
        if self.complex:
            #print("Ik zit in de forward")
            encoded_batch_real = x.real
            encoded_batch_imag = x.imag
            #print(encoded_batch_real.shape)
            #print(encoded_batch_imag.shape)
            intermediate_real, intermediate_imag = complex_conv(encoded_batch_real, encoded_batch_imag, self.Conv1_real, self.Conv1_imag)
            #print(intermediate_real.shape)
            intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, self.device), complex_batchnorm(intermediate_imag, self.device)
            #print(intermediate_real.shape)

            intermediate_real, intermediate_imag = complex_relu(intermediate_real, self.device), complex_relu(intermediate_imag, self.device)

            intermediate_real, intermediate_imag = complex_conv(intermediate_real, intermediate_imag, self.Conv2_real, self.Conv2_imag)
            intermediate_real, intermediate_imag = complex_batchnorm(intermediate_real, self.device), complex_batchnorm(intermediate_imag, self.device)

            #print(intermediate_real.shape)
            #print(intermediate_imag.shape)
            if self.downsample_conv_real is not None:
                encoded_batch_real, encoded_batch_imag = complex_conv(encoded_batch_real, encoded_batch_imag, self.downsample_conv_real, self.downsample_conv_imag)


            #print("HOI", intermediate_real.shape)
            #print(intermediate_imag.shape)

            intermediate_real = intermediate_real + encoded_batch_real
            intermediate_imag = intermediate_imag + encoded_batch_imag

            intermediate_real, intermediate_imag = complex_relu(intermediate_real, self.device), complex_relu(intermediate_imag, self.device)

            x = torch.complex(intermediate_real, intermediate_imag)
            #print("Ik ga uit de forward")
            return x



        else:    
            z = self.net(x)
            if self.downsample is not None:
                x = self.downsample(x)
            out = z + x
            out = self.act_fn(out)
            return out

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device

