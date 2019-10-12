import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import utils.d2lzh as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_input = 784
num_output = 10

class LinearNet(nn.Module):
    def __init__(self, num_input, num_output):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_input, num_output)
    def forword(self, X):
        y = self.linear(X.view(X.shape[0], -1))
        return y

FlattenLayer = d2l.FlattenLayer

net = nn.Sequential()
net.add_module('flatten', FlattenLayer())
net.add_module('linear', nn.Linear(num_input, num_output))

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

print(net)
print(net[1].weight,net[1].bias)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epoch = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, batch_size, None, None, optimizer)

X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])