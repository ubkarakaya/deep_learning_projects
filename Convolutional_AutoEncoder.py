import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from Utility import*
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # ENCODER LAYERS
        # conv layer
        self.conv1 = nn.Conv2d(1, 100, 28, padding=1)
        # conv layer
        self.conv2 = nn.Conv2d(100, 100, 1, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # DECODER LAYERS
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(100, 100, 1, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(100, 100, 1, stride=2)

    def forward(self, x):
        # ENCODE
        # add hidden layers with relu activation function
        # and max-pooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        # DECODE
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x


def test(model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, _ = data
            outputs = model(images)
            # sum up batch loss
            test_loss += loss_function(outputs, images)

    test_loss /= len(test_loader.dataset)
    print('-----> Test Loss: {:.8f}'.format(test_loss))


def run_cae(train_loader, test_loader):
    # initialize the model
    model = ConvAutoencoder()
    # selection of the loss function
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # number of epochs to train the model
    n_epochs = 10

    for epoch in range(1, n_epochs + 1):

        train_loss = 0.0
        # TRAINING STEP
        for data in train_loader:
            # _ stands in for labels, here
            images, _ = data
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss
            # print avg training statistics
        train_loss /= (len(train_loader) * n_epochs)
        print('Epoch: {} \t Training Loss: {:.8f}'.format(epoch, train_loss))

        test(model, test_loader, criterion)

    return model
