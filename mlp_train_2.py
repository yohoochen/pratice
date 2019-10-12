import torch
from torch import nn
from torch.nn import init
import numpy as np
import utils.d2lzh as d2l

#shuju
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#moxing
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential()
net.add_module('FLatten', d2l.FlattenLayer())
net.add_module('linear1', nn.Linear(num_inputs, num_hiddens))
net.add_module('ReLu', nn.ReLU())
net.add_module('linear2', nn.Linear(num_hiddens, num_outputs))

# net2 = nn.Sequential( d2l.FlattenLayer(),
#         nn.Linear(num_inputs, num_hiddens),
#         nn.ReLU(),
#         nn.Linear(num_hiddens, num_outputs),
#         )

#init net
for param in net.parameters():
    init.normal_(param, mean=0, std=0.01)
    # param.requires_grad_(requires_grad=True)

#loss&optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

#epoch
num_epochs = 5

#train
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)