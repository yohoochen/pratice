import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, n_feature):
        super(Linear, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    def forward(self, x):
        y = self.linear(x)
        return y


if __name__ == "__main__":
    net = nn.Sequential()
    net.add_module("linear", nn.Linear(2,1))
    # print(net)
    # print(net[0])

    linear = Linear(2)
    print(linear)
    net.add_module("linear2", linear)
    # print(net)
    # print(net[1])