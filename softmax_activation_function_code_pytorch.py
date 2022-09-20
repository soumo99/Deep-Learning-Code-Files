
# -------- APPLYING SOFTMAX ACTIVATION FUNCTION ----------

import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x) , axis = 0) #dividing the exponentials with the sum of exponentials

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy : ',outputs)

x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x , dim = 0)#dimensions = 0 so that it computes along with the first axis
print('pytorch tensor value : ',outputs)


# -----------CALCULATING CROSS ENTROPY LOSS ----------------

import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]

Y = np.array([1,0,0])

# Y pred has probabilities
Y_pred_good = np.array([0.7 , 0.2 , 0.1])
Y_pred_bad = np.array([0.1 , 0.3 , 0.6])

loss_1 = cross_entropy(Y,Y_pred_good)
loss_2 = cross_entropy(Y,Y_pred_bad)

print(f'Loss1 numpy : {loss_1 :.4f}')
print(f'Loss2 numpy : {loss_2 :.4f}')


# using pytorch

loss = nn.CrossEntropyLoss()

Y =  torch.tensor([0]) # class 0
# size = number of samples * number of classes  --> 1sample *  3 classes = 3
Y_pred_good = torch.tensor([[2.0, 1.0 , 0.1]]) # raw values , softmax not applied
Y_pred_bad = torch.tensor([[0.5, 2.0 , 0.3]])

#calculating the loss
loss_1 = loss(Y_pred_good,Y)
loss_2 = loss(Y_pred_bad,Y)

print(loss_1.item())
print(loss_2.item())

# Now checking the actual predictions value

_,predictions_1 = torch.max(Y_pred_good,1)
_,predictions_2 = torch.max(Y_pred_bad,1)

print("The first predictions : ",predictions_1)
print("The second predictions : ",predictions_2)


# Checking with 3 samples

Y =  torch.tensor([2, 0, 1])
# size = number of samples * number of classes  --> 3sample *  3 classes = 9
Y_pred_good = torch.tensor([[0.1, 1.0 , 2.1],[2.0, 1.0 , 0.1],[0.1, 3.0 , 0.1]]) # raw values , softmax not applied
Y_pred_bad = torch.tensor([[2.1, 1.0 , 0.1],[0.1, 1.0 , 2.1],[0.1, 3.0 , 0.1]])

#calculating the loss
loss_1 = loss(Y_pred_good,Y)
loss_2 = loss(Y_pred_bad,Y)

print(loss_1.item())
print(loss_2.item())

# Now checking the actual predictions value

_,predictions_3 = torch.max(Y_pred_good,1)
_,predictions_4 = torch.max(Y_pred_bad,1)

print("The first predictions : ",predictions_3)
print("The second predictions : ",predictions_4)