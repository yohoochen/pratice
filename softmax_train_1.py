import torch
import torchvision
import numpy as np
from utils.d2lzh import *

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.size())
# print(X.sum(dim=0, keepdim=True).size())

def soft_max(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

#test softamx
# input = torch.rand(5, 6)
# s = soft_max(input)
# print(input,"\n", s, "\n", s.sum(dim=1))

def net(X):
    return soft_max(torch.mm(X.view(-1,num_inputs), W) + b)

#learn gather()
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])
# print(y_hat.gather(1, y.view(-1, 1)))
# print(y.view(-1, 1))
# def accuracy(y_hat, y):
#     return (y_hat.argmax(dim=1)==y).float().mean().item()
# print(accuracy(y_hat, y))


num_epochs, lr = 5, 0.1
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
X, y = iter(test_iter).next()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])