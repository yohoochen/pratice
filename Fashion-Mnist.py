import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys

# sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import utils.d2lzh as d2l

# print(torch.__version__)
# print(torchvision.__version__)
mnist_train = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./dataset', train=False, download=True, transform=transforms.ToTensor())
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))

feature, lable = mnist_train[0]
print(feature.shape, lable)

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))