#Pseudocode of the complex lenet model

class EncoderGenerator(nn.Module):

    def __init__(self, k, device):
        # Define first layers of network

    def forward(self, a)
       

class EncoderDiscriminator(nn.Module):

	def __init__(self):
        # Define linear layer
        # Define sigmoid

	def forward(self,generated, a):
        # Input: a and a' (maybe shuffled)
        # Apply linear layer
        # Apply sigmoid
        # Output: decision if fake (0) or real (1) (maximize this so it does not know which one is which)

class LenetEncoder(nn.Module):
    def __init__(self, k, device):
        # Define g function part of model
        # Define discriminator part of model

    def forward(self, x):
        # Input: batch with images

        # Learning part
        # Apply first layers of network and put it in variable a
        # Send a to discriminator
        # Send a to transform function and get transformed a'
        # Send a and a' to discriminator

        #Sent through part
        # Transform with angle and complex value b
        # Output: transformed image, theta for decoder

    def transform(self, a):
        # Input: a
        # Make it complex
        # Multiply with angle
        # Make it real again
        # return transformed a        

class LenetProcessingModule(nn.Module):
    def __init__(self, device):
        # Define all other layers (helaas complex)
    
    def forward(self, x):
        # Input: encoded image
        # Go through most of the layers in complex space
        # Output: encoded image

class LenetDecoder(nn.Module):
    def __init__(self):
        # Define last layers

    def forward(self, x, theta):
        # Input: encoded image, theta
        # Rotate back
        # Get real values
        # Make decision with crossentropy


class ComplexLenet(nn.Module):
    def __init__(self, device, k=5 ):
        # Define device
        # Define connection to encoder
        # Define connection to processing module
        # Define connection to decoder

    def forward(self, x):
        # Input: batch with images
        # Forward in encoder
        #   - receive encoded image for processing module
        #   - sigmoid output discriminator
        #   - theta for decoder
        # Forward in processing module
        #   - receive encoded image for decoder module
        # Forward in decoder
        #   - receive crossentropyloss with softmax