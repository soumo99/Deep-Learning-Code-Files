import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0 , 1.0 , 2.0 , 3.0])

# SOFTMAX ACTIVATION FUNCTION

output = torch.softmax(x ,dim=0)
print(output)
sm = nn.Softmax(dim = 0)
output = sm(x)
print(output)

# SIGMOID ACTIVATION FUNCTION

output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)

# TANH ACTIVATION FUNCTION

output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)

# RELU ACTIVATION FUNCTION

output = torch.relu(x)
print(output)
r = nn.ReLU()
output = r(x)
print(output)

# LEAKY RELU ACTIVATION FUNCTION

output = F.leaky_relu(x)
print(output)
l_relu = nn.LeakyReLU()
output = l_relu(x)
print(output)


#--------------------------------------------------------------------------------------------------------------------

# ---------- PROCESS 1 ----------------- Create nn Module

class act_1(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(act_1,self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size) # first layer
        self.relu  = nn.ReLU() # Activation function
        self.linear_2 = nn.Linear(hidden_size, 1) # Last layer
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.Sigmoid(out)

        return out


# ------------------ PROCESS 2 ---------------- Use activation function directly in forward pass

class act_2(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(act_2,self).__init__()
        self.linear_1 = nn.Linear(input_size,hidden_size) # first layer
        self.linear_2 = nn.Linear(hidden_size, 1) # Last layer

    def forward(self,x):
        out = torch.relu(self.linear_1(x))
        out = torch.sigmoid(self.linear_2(out))

        return out
