import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784
hidden_size = 100
output_size = 10
num_epochs = 2
batch_size = 100 # num of training examples utilized in one iteration
learning_rate = 0.001

# MNIST DATASET

train_dataset = torchvision.datasets.MNIST(root = './pytorch_codes',train = True,
                                           transform=transforms.ToTensor(),download=True)

test_dataset = torchvision.datasets.MNIST(root = './pytorch_codes',train = False,
                                           transform=transforms.ToTensor())

# Creating the dataloader

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size ,
                                           shuffle = False)



examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)


# Plotting
for i in range(6):
    plt.subplot(2, 3, i+1) # (rows, cols, index+1)
    plt.imshow(samples[i][0], cmap = 'gray')

#plt.show()

class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.linear_1 = nn.Linear(input_size , hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)

        return out


# MODEL
model = NeuralNet(input_size , hidden_size , output_size)

# LOSS AND OPTIMIZER

criterion = nn.CrossEntropyLoss() # It applies a softmax function
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# Training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # enumerate function will give the actual index and the data i.e tuples of images and labels
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward Pass

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward Pass
        optimizer.zero_grad() # to empty the gradients of the zero attribute
        loss.backward() # for backpropagation
        optimizer.step() # for update step


        if (i+1) % 100 == 0: # for every 100 iterations it will print some value
            print(f'epoch {epoch + 1} / step{i+1}/{n_total_steps} , loss = {loss.item() :.4f} ')


        # Testing and Evaluation
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.reshape(-1,28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)

                # value, index
                _,predicted = torch.max(outputs, 1)
                n_samples += labels.size(0) # gives the number of samples in the current batch
                n_correct += (predicted == labels).sum().item()

            # Total accuracy
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images : {acc} ')

