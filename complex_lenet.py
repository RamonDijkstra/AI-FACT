import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, k):
        """
        """
        super().__init__()
        self.b = torch.rand()
        
        self.thetas = nn.Parameter(torch.empty(k-1))
        self.thetas = toch.rand(k-1) * torch.pi


    def forward(self, a):
        """
        """

        x = self.layers(z)
        x = x.view(z.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2])
        
        return x

    @property
    def device(self):
        """
        """
        return next(self.parameters()).device


class Discriminator(nn.Module):

    def __init__(self, input_dims=784, hidden_dims=[512, 256], dp_rate=0.3):
        """
        """
        super().__init__()


    def forward(self, x):
        """
        """
        preds = self.layers(x)

        return preds

class Lenet_encoder(nn.Module):
    def __init__(self, k=4):
        super(Lenet_encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.k = k
        self.generator = Generator()

    def forward(self, x):
        # conv1 is the encoder which maps the input to the feature 
        a = self.conv1(x)
        # We use the feature to generate k - 1 fake features
        k = self.generator(a)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Lenet_processing_module(nn.Module):
    def __init__(self):
        super(Lenet_processing_module, self).__init__()
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        pass

class Lenet_decoder(nn.Module):
    def __init__(self):
        super(Lenet_decoder, self).__init__()

    def forward(self, x):
        pass
        