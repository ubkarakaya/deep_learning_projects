import torch
from mxnet import gluon
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import Convolutional_AutoEncoder as cna
from sklearn.model_selection import KFold
import DenosingConvolutional_AutoEncoder as dca
import Variational_AutoEncoder as vae
import warnings
warnings.filterwarnings('ignore')
import fine_tuning as ft
import torchvision
import torch.onnx as onnx
import torchvision.models as models

if __name__ == '__main__':

    # Initialization of Batch Size
    bs = 100
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Create train and test data loaders
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # Cross Validation of train dataset
    kf = KFold(n_splits=2)
    kf.get_n_splits(train_dataset)
    for train_index, val_index in kf.split(train_dataset):

        # Cross- Validation
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(val_index)
        # Split the dataset according to indices
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, sampler=train_sampler, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, sampler=valid_sampler, num_workers=0)

        # Convolutional AutoEncoder
        model = cna.run_cae(train_loader, test_loader)
        # Denosing Convolutional AutoEncoder
        #dca.run_cae(train_loader, test_loader)
        # Variational Convolutional AutoEncoder
        #vae.run_vae(train_loader, test_loader)
        #  fine-tuning
        # The model parameters in the output layer will be iterated using a learning
        # rate ten times greater

        ft.train_fine_tuning(model, train_loader, test_loader)



