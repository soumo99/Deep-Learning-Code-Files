''' Convolution Neural Network or neural nets are mainly the ordinary nets made up of
neurons that have learnable weights and biases and mainly works on image data and apply the so
 called convolution filters '''

# torch.nn means neural network from scratch

from torch.nn import Module  # base class used to develop all neural network models.
from torch.nn import Conv2d  # for convolutional layers
from torch.nn import Linear  # fully connected layer
from torch.nn import MaxPool2d  # Applies 2D max-pooling to reduce the spatial dimensions of the input volume
from torch.nn import ReLU  # activation function
from torch.nn import LogSoftmax  # Used when building our softmax classifier to return the predicted probabilities of each class
from torch import flatten # Flattens the output of a multi-dimensional volume


class LeNet(Module):
    def __init__(self,num_channels,classes): # num_channels - number of channels in the input images (1 for grayscale and 3 for RGB) and classes - Total number of unique class labels in our dataset
        super(LeNet,self).__init__()

        # Initializing the first Conv --> Relu --> Pool Layer
        # kernel is a  filter that is used to extract the features from the images.
        self.conv_1 = Conv2d(in_channels=num_channels, out_channels=20
                             , kernel_size=(5,5))
        self.relu_1 = ReLU()
        self.maxpool_1 = MaxPool2d(kernel_size=(2,2),stride=(2,2)) # stride is used to reduce the spatial dimensions of the input image

        # Initializing the second Conv --> Relu --> Pool Layer
        # kernel is a  filter that is used to extract the features from the images.
        self.conv_2 = Conv2d(in_channels=20, out_channels=50
                             , kernel_size=(5, 5))
        self.relu_2 = ReLU()
        self.maxpool_2 = MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2))  # stride is used to reduce the spatial dimensions of the input image


        # first set of fully connected layers
        self.fully_con_1 = Linear(in_features=800,out_features=500)
        self.relu_3 = ReLU()

        self.fully_con_2 = Linear(in_features=500,out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1) # logsoftmax used such that we can obtain predicted probabilities during evaluation

        # Building the first layer
        # Passing the input through the first set of CONV --> RELU -->
        def forward(self, x):
            x = self.conv_1(x)
            x = self.relu_1(x)
            x = self.maxpool_1(x)

        # Building the second layer
        # Passing the input through the first set of CONV --> RELU --> POOL layers
            x = self.conv_2(x)
            x = self.relu_2(x)
            x = self.maxpool_2(x)

        # x is a multi-dimensional tensor
        # FLatten the output from the previous layer and pass it
        # Flatten is reuqired to create a fully connected layer
        # Fully connected 1 and relu is being connected to the network architecture
            x = flatten(x,1)
            x = self.fully_con_1(x)
            x = self.relu_3(x)

            # Finally the fully connected layer 2 and softmax is being connected
            # Pass the output to the softmax classifier to get the output
            x = self.fully_con_2(x)
            output = self.logSoftmax(x)


            # Return the output predictions
            return output # The output is then returned to the calling function
