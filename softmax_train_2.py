import torch
from torch import nn
from torch.nn import init
import numpy as np
import utils.d2lzh as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_input = 784
num_output = 10

class LinearNet(nn.module):
    def __init__(self, num_input, num_output):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_input, num_output)
    def forword(self, X):
        y = self.linear(X.view(X.shape[0], -1))
        return y

