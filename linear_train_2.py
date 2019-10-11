import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
from model.Linear import Linear
from dataset.Linear_dataset import getdata


num_input = 2
net = nn.Sequential()
# linear = Linear(num_input)
# net.add_module("linear", linear)
net.add_module("linear2", nn.Linear(num_input, 1))

dataset = getdata()

#初始化模型参数
print(net)
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
# print(net[0])

#初始化损失函数
loss = nn.MSELoss()

#定义优化器
optimizer = optim.SGD(net.parameters(), lr= 0.03)
print(optimizer)

#开始训练
epoch_num = 3
for epoch in range(epoch_num):
    for X, y in dataset:
        output = net(X)
        l = loss(output, y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

true_w = [2, -3.4]
true_b = 4.2
dense = net[0]

print(true_w, dense.weight)
print(true_b, dense.bias)

