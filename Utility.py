import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.utils import save_image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_graph(data, labels, fname):
    digit_to_color = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                      "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    xs = np.array([x[0] for x in data])
    ys = np.array([x[1] for x in data])

    fig, ax = plt.subplots()
    labels_to_show = labels[0:len(data)]
    for digit in range(10):
        ix = np.where(labels_to_show == digit)
        ax.scatter(xs[ix], ys[ix], c=digit_to_color[digit],
                   label=digit, marker=".")
    ax.legend()
    plt.savefig(fname)
    plt.show()


# 28x28 = 784 Pixels to show points

def flatten_input(train_dl, MNIST_NUM_PIXELS=784):
    flat_input = []
    labels = []
    for features, targets in train_dl:
        flat_input.append(features.view(-1, MNIST_NUM_PIXELS).detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
    return np.concatenate(flat_input), np.concatenate(labels)


def visualization(train_loader, test_loader):
    flat_test_input, test_labels = flatten_input(test_loader)
    flat_train_input, train_labels = flatten_input(train_loader)

    pca = PCA(n_components=2, random_state=10)
    pca.fit(flat_train_input)
    transformed = pca.transform(flat_test_input[0:5000])
    plot_graph(transformed, test_labels, "pca_repr.png")
