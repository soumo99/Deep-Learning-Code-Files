import torch
import torch.nn as nn

# Multiclass problem

class Multi(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Multi,self).__init__()
        self.linear_1 = nn.Linear(input_size,hidden_size) # first layer
        self.relu  = nn.ReLU() # Activation function
        self.linear_2 = nn.Linear(hidden_size,output_size) # Last layer

    def forward(self,x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)

        return out

model = Multi(input_size=28 *28, hidden_size=5, output_size=3)
criterion  = nn.CrossEntropyLoss()