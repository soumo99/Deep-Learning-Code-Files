import torch

# creating empty tensors with size 3
x = torch.empty(3)
print(x)

# size 2,3 matrix
x = torch.empty(2,3)
print(x)

# creating tensors with random numbers
y = torch.rand(2,2)
print(y)

# creating matrix with zeros
a = torch.zeros(2,2)
print(a)

# creating matrix with ones
a = torch.ones(2,2, dtype = torch.float64)
print(a.size())


# printing tensors
x = torch.tensor([2.5,1.2])
print(x)

# --------BASIC OPERATION-------

x = torch.rand(2,2)
y = torch.rand(2,2)
z = x+y
z = torch.add(x,y)
print(z)

# trailing underscore performs inplace operations (_)
# Basically it adds all the values of x to y

x = torch.rand(2,2)
y = torch.rand(2,2)
y.add_(x)
print(y)


# Subtraction
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x-y
z = torch.sub(x,y)
print(z)

# Multiplication
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x*y
z = torch.mul(x,y)
print(z)

y.mul_(x)
print(y)

y.mm(x)
print(y)

m = x@y
print(m)



# Division
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x/y
z = torch.div(x,y)
print(z)


# ---------Slicing operations------------
x = torch.rand(5,3)
print(x)
print(x[0,:-1])
print(x[-1])
print(x[:,:-1])

print(x[1,1].item()) #for getting the actual value


# -------Reshaping Tensors----------
x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)

y = x.view(-1,8)
print(y.size())


# -----converting from numpy arrays to torch tensors--------
import torch
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# shares the same memory location
a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)

# This can be prevented using torch.tensors

tensors = torch.tensor(a)
tensors += 1
print(tensors)
print(a)


