import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_fine_tuning(naive_dae, train_dl_flat, test_loader):
    print('fine-tuning')
    # fine-tune autoencoder
    lr = 1e-3
    loss = nn.MSELoss()
    optimizer = optim.Adam(naive_dae.parameters(), lr)
    num_epochs = 10

    # train
    running_loss = float("inf")
    for epoch in range(num_epochs):
        losses = []
        for i, data_list in enumerate(train_dl_flat):
            data = data_list[0]
            v_pred = naive_dae(data)
            batch_loss = loss(data, v_pred)  # difference between actual and reconstructed
            losses.append(batch_loss.item())
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        running_loss = np.mean(losses)
        print(f"Epoch {epoch}: {running_loss}")
