import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import Utility as ut
from torchvision.utils import save_image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def make_dir():
    image_dir = 'Saved_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def save_decoded_image(img, name):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, name)


#  Auto Encoder Network
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x)  # the latent space representation

        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.out(x))
        return x


# the training function
def train(device, net, trainloader, optimizer, num_epochs, noise_factor, criterion):
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data  # we do not need the image labels
            # add noise to the image data
            img_noisy = img + noise_factor * torch.randn(img.shape)
            # clip to make the values fall between 0 and 1
            img_noisy = np.clip(img_noisy, 0., 1.)
            img_noisy = img_noisy.to(device)
            optimizer.zero_grad()
            outputs = net(img_noisy)
            loss = criterion(outputs, img_noisy)
            # backpropagation
            loss.backward()
            # update the parameters
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, num_epochs, loss))
        save_decoded_image(img_noisy.cpu().data, name='./Saved_Images/noisy{}.png'.format(epoch))
        save_decoded_image(outputs.cpu().data, name='./Saved_Images/denoised{}.png'.format(epoch))
    return train_loss


def test_image_reconstruction(device, net, testloader, criteration, noise_factor):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, _ = data
            outputs = net(images)
            # sum up batch loss
            test_loss += criteration(outputs, images)

    test_loss /= len(testloader.dataset)
    print('-----> Test Loss: {:.8f}'.format(test_loss))

    for batch in testloader:
        img, _ = batch
        img_noisy = img + noise_factor * torch.randn(img.shape)
        img_noisy = np.clip(img_noisy, 0., 1.)
        img_noisy = img_noisy.to(device)
        outputs = net(img_noisy)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(img_noisy, 'noisy_test_input.png')
        save_image(outputs, 'denoised_test_reconstruction.png')
        break


def run_cae(train_loader, test_loader):
    # Fine-tune
    # constants
    num_epochs = 2
    learning_rate = 1e-3
    noise_factor = 0.5

    net = Autoencoder()
    print(net)

    # the loss function
    criterion = nn.MSELoss()
    # the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    device = get_device()
    print(device)
    net.to(device)
    make_dir()
    train_loss = train(device, net, train_loader, optimizer, num_epochs, noise_factor, criterion)
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./Saved_Images/conv_ae_mnist_loss.png')
    test_image_reconstruction(device, net, test_loader, criterion, noise_factor)
    ut.visualization(train_loader, test_loader)
