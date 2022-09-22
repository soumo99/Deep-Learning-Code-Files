import torch
import torch.nn as nn

# Binary class  problem

class Binary(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Binary,self).__init__()
        self.linear_1 = nn.Linear(input_size,hidden_size) # first layer
        self.relu  = nn.ReLU() # Activation function
        self.linear_2 = nn.Linear(hidden_size,1) # Last layer

    def forward(self,x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)

        # Sigmoid function at the end
        y_pred = torch.sigmoid(out)
        return y_pred


model = Binary(input_size=28 *28, hidden_size=5, output_size=3)
criterion  = nn.BCELoss()