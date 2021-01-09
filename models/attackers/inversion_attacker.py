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
Model for inversion attacks
"""

import torch
import torchvision
import torchvision.transforms as transforms

class DoubleConv(nn.Module):
    """
	Double convolution layer. 
	Adapted from https://github.com/milesial/Pytorch-UNet
	"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
		
class Down(nn.Module):
    """
	Downscaling with maxpool then double convolution.
	Adapted from https://github.com/milesial/Pytorch-UNet
	"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
	Upscaling then double convolution.
	Adapted from https://github.com/milesial/Pytorch-UNet
	"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
		
class OutConv(nn.Module):
	"""
	Output convolution.
	Adapted from https://github.com/milesial/Pytorch-UNet
	"""
	
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class InversionAttacker(nn.Module):
	"""
	Inversion attack model
	"""

    def __init__(self, encoded_feature_shape=[1, 28, 28], image_shape=[1, 28, 28]):
        """
        Network for inversion attacks based on the U-net.

        Inputs:
			encoded_feature_shape - Shape of the encoded feature. The model will try to 
				reconstruct the original feature from the encoded feature.
			image_shape - Shape of the original images. The output will be a reconstructed image.
        """
        super().__init__()
		
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

    @property
    def device(self):
        """
        Property function to get the device on which the model is
        """
        return next(self.parameters()).device