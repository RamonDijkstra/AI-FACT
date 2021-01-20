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

        # initialize the different modules of the network
        self.encoder = GAN(self.k, self.lr, num_blocks[0])
        self.proccessing_module = Resnet_Processing_module(num_blocks[1])
        self.decoder = Resnet_Decoder(num_blocks[2], self.num_classes)
        self.softmax = nn.Softmax()
        
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

        # # send the encoded feature to the processing unit
        # out = self.proccessing_module(out)
        
        # # decode the feature from
        # result = self.decoder(out, thetas)

        # # Log the train accuracy
        # out = self.softmax(result)
        # preds = out.argmax(dim=-1)
        # acc = (labels == preds).float().mean()
        # self.log('train_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards
        
        # # return the decoded feature, discriminator predictions and real labels
        # # return x, discriminator_predictions, labels
        # model_loss = self.loss_fn(result, labels)
        
        # loss = gan_loss + model_loss

        # # log the loss
        # self.log("generator/loss", gan_loss)
        # self.log("model/loss", model_loss)
        # self.log("total/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, labels = batch

        # run the image batch through the encoder (generator and discriminator)
        gan_loss, out, thetas = self.encoder(x, False)
        
        # send the encoded feature to the processing unit
        out = self.proccessing_module(out)
        
        # decode the feature from
        result = self.decoder(out, thetas)
        result = self.softmax(result)
        preds = result.argmax(dim=-1)
        acc = (labels == preds).float().mean()

        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards

class GAN(nn.Module):

    def __init__(self, k, lr, num_blocks):
        """
        GAN model used in the encoder of the complex networks

        Inputs:
            k - Level of anonimity. k-1 fake features are generated
                to train the discriminator. Default = 2
            lr - Learning rate to use for the optimizer. Default = 1e-3
        """
        super().__init__()
        
        # save inputs
        self.k = k
        self.lr = lr

        # initialize the generator and discriminator
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        self.generator = EncoderGenerator(k, device, num_blocks)
        self.discriminator = EncoderDiscriminator(device)

    def configure_optimizers(self):
        """
        Function to configure the optimizers
        """
        
        # initialize optimizer for both generator and discriminator
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        # return the optimizers
        return [optimizer_gen, optimizer_disc], []

    def forward(self, x, optimizer_idx):
        """
        """
        
        # include the discriminator if the optimizer_idx=1 (discriminator optimizer)
        if optimizer_idx == 1:
            loss, x, thetas = self.discriminator_step(x)
        else:
            loss, x, thetas = self.generator_step(x)
        
        # return the loss, real encoded feature and real angles
        return loss, x, thetas

    def generator_step(self, x):
        """
        """

        # encode the convolved images using the generator
        with torch.no_grad():
            a, x, thetas, fake_x, delta_thetas = self.generator(x)
        
        # create labels for the fake features
        labels_fake = torch.ones([fake_x.shape[0], 1],requires_grad=True).to(self.device)
        
        # let the discriminator predict the fake fatures
        preds = self.discriminator(fake_x)
        
        # compute the loss over the fake features
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds, labels_fake)

        # return the generator loss, real encoded feature and real angles
        return loss, x, thetas

    def discriminator_step(self, x):
        """
        """

        # encode the convolved images using the generator
        with torch.no_grad():
            a, x, thetas, fake_x, delta_thetas = self.generator(x)

        # create a batch of real feature and encoded fake features
        real_and_fake_images = torch.cat([a, fake_x], dim=0).to(self.device)
        
        # create the labels for the batch
        # 1 if real, 0 if fake
        labels = torch.cat([torch.ones(a.shape[0]),torch.zeros(fake_x.shape[0])], dim=0)
        labels = labels.reshape(labels.shape[0], 1).to(self.device)
        
        # predict the labels using the discriminator
        discriminator_predictions = self.discriminator(real_and_fake_images)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(discriminator_predictions, labels)

        # return the discriminator loss, real encoded feature and real angles
        return loss, x, thetas
        
    @property
    def device(self):
        """
        Property function to get the device on which the GAN is
        """
        
        return next(self.parameters()).device

class EncoderGenerator(nn.Module):
    """
	Generator part of the LeNet encoder model
	"""
    
    def __init__(self, k, device, num_blocks):
        """
        Generator model of the encoder

        Inputs:
            k - Level of anonimity. k-1 fake features are generated
                to obscure the real feature.
            device - PyTorch device used to run the model on.
        """
        super().__init__()

        # save the inputs
        self.k = k
        self.device = device
        self.filter_0 = 16 #this comes from c_hidden[0]

        self.input_net = nn.Sequential(
                nn.Conv2d(3, self.filter_0, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.filter_0),
                nn.ReLU()
            )

        # Creating the ResNet blocks
        blocks = []
        for i in range(num_blocks):
            subsample = (i == 0)
            blocks.append(
                    ResNetBlock(c_in=self.filter_0,
                                             act_fn=nn.ReLU(),
                                             subsample=subsample,
                                             c_out=self.filter_0)
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, training=True):
        """
        Inputs:
            a - Input batch of convolved images. Shape: [B, C, W, H]
                B - batch size
                C - channels per image
                W- image width
                H - image height
            training - Boolean value. Default = True
                True when training
                False when using in application
        Outputs:
            a - Real non-encoded feature. Shape: [B, C, W, H]
            x - Real encoded feature. Shape: [B, C, W, H]
            thetas - Angles used for rotating the real feature. Shape: [B, 1, 1, 1]
            fake_x - Fake generated features. Shape: [B * k-1, C, W, H]
            delta_thetas - Angles used for rotating the fake features. Shape: [B * k-1, 1, 1, 1]
        """

        a = self.input_net(x).to(self.device)
        a = self.blocks(a)
        # save the image dimensions for later use
        image_dimensions = a.shape
        
        # compute the magnitude of the image batch
        a_magnitude = torch.norm(torch.norm(a, dim=(2,3), keepdim=True), dim=1, keepdim=True)

        # create real obfuscating features b
        b = torch.normal(0, 1, size=tuple((image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3])))
        b = b.to(self.device)
        # b_magnitude = torch.sqrt(torch.sum(torch.square(b)).type(torch.FloatTensor))
        b_magnitude = torch.norm(torch.norm(b, dim=(2,3), keepdim=True), dim=1, keepdim=True)
        b = (b / b_magnitude) * a_magnitude

        # sample angles to rotate the features for the real rotation
        thetas = torch.Tensor(image_dimensions[0], 1, 1, 1).uniform_(0, 2 * np.pi).to(self.device)
        thetas = torch.exp(1j * thetas)
        
        # compute encoded real feature
        x = (a + b * 1j) * thetas
        x = x.to(self.device)
        
        # check if training
        if training:      
            # create fake obfuscating features b
            fake_b = torch.normal(0, 1, size=tuple(((self.k-1) * image_dimensions[0], image_dimensions[1], image_dimensions[2], image_dimensions[3]))).to(self.device)
            fake_b_magnitude = torch.norm(torch.norm(fake_b, dim=(2,3), keepdim=True), dim=1, keepdim=True)
            fake_b = (fake_b / fake_b_magnitude)* a_magnitude
        
            # sample k-1 delta angles to rotate the features for fake examples
            delta_thetas = torch.Tensor((self.k-1) * image_dimensions[0], 1, 1, 1).uniform_(0, np.pi).to(self.device)
            delta_thetas = torch.exp(1j * delta_thetas)
        
            # compute encoded fake features
            fake_a = torch.cat([a]*(self.k-1),dim=0)
            fake_x = (fake_a + fake_b *1j) * delta_thetas
            fake_x = fake_x.to(self.device)
            
            # return real feature, real encoded feature, thetas, fake encoded feature and delta thetas
            return a, x, thetas, fake_x, delta_thetas
        else:
            # return the real encoded features and thetas
            return x, thetas

class EncoderDiscriminator(nn.Module):
    """
	Discriminator part of the LeNet encoder model
	"""    
    def __init__(self, device):
        """
        Discriminator model of the encoder
        """
        super().__init__()
        
        self.device = device

        # initialize the linear layer
        self.linear = nn.Linear(16*32*32,1)
        
        # initialize the sigmoid layer
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoded_batch):
        """
        Inputs:
            encoded_batch - Input batch of encoded features. Shape: [B, C, W, H]
                B - batch size
                C - channels per feature
                W- feature width
                H - feature height
        Outputs:
			predictions - Predictions for real and fake feature. Shape: [B, 1]
                1 when real feature
                0 when fake feature
        """

        #to real values
        encoded_batch = encoded_batch.real
        
        encoded_batch_shape = encoded_batch.shape[1] * encoded_batch.shape[2] * encoded_batch.shape[3]
        # reshape the batch
        encoded_batch = encoded_batch.view(encoded_batch.shape[0],encoded_batch_shape).to(self.device)
        # predict the labels
        predictions = self.linear(encoded_batch)
        predictions = self.sigmoid(predictions)
        
        # return the predictions
        return predictions

### Complex ResNet alpha variant:
### - Encoder: All layers before 16x16 (so the first 1+2n blocks)
### - Proc module: All 32x32 blocks + the first 8x8 block)
### - Decoder: All but the first 8x8 blocks

class Resnet_Processing_module(nn.Module):
    def __init__(self,  num_blocks):

        super().__init__()
        self.num_blocks = num_blocks

        blocks = []     # Creating the ResNet blocks -- Normal for now, need to make complex layers
        for i in range(self.num_blocks):
            subsample = (i == 0)
            blocks.append(
            ResNetBlock(c_in=32,
                            act_fn=nn.ReLU(),
                            subsample=subsample,
                            c_out=32)
                    )
            blocks.append(
                ResNetBlock(c_in=64,
                                    act_fn=nn.ReLU(),
                                    subsample=True,
                                    c_out=64))
            self.blocks = nn.Sequential(*blocks)

    def forward(self, encoded_batch):
        pass




class Resnet_Decoder(nn.Module):
    def __init__(self, num_blocks, num_classes):

        super().__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        blocks = []
        for i in range(self.num_blocks-1):
            blocks.append(
                    ResNetBlock(c_in=64,
                                    act_fn=nn.ReLU(),
                                    subsample=False,
                                    c_out=64)
                    )
        
        self.blocks = nn.Sequential(*blocks)
        
        self.output_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(64, self.num_classes)
            )

    def forward(self,x, thetas):
    #Rerotate
    #Make real

    # Go through Blocks and output net
        pass


class ResNet(nn.Module):

    def __init__(self, num_classes=10, num_blocks=[3,3,3], c_hidden=[16,32,64], act_fn_name="relu", block_name="ResNetBlock", **kwargs):
        """
        Inputs: 
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace(num_classes=num_classes, 
                                       c_hidden=c_hidden, 
                                       num_blocks=num_blocks, 
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name],
                                       block_class=resnet_blocks_by_name[block_name])
        self._create_network()
        self._init_params()

        
    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        
        # A first convolution on the original image to scale up the channel size
        
        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
                blocks.append(
                    self.hparams.block_class(c_in=c_hidden[block_idx if not subsample else (block_idx-1)],
                                             act_fn=self.hparams.act_fn,
                                             subsample=subsample,
                                             c_out=c_hidden[block_idx])
                )
        self.blocks = nn.Sequential(*blocks)
        
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )
        
    
    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x

class ResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn=nn.ReLU(), subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()

        if not subsample:
            c_out = c_in
            
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
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out

class PreActResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in
            
        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        )
        
        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False)
        ) if subsample else None

        
    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out

resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock
}

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}