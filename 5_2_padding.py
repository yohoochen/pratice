import torch
from torch import nn


def comp(conv2d, X):
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels=1, out_channels=1, padding=1, kernel_size=3)

X = torch.rand(8, 8)
Y = comp(conv2d, X)
print(Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp(conv2d, X).shape)
