import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils.d2lzh as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('dataset/kaggle_house/train.csv')
test_data = pd.read_csv('dataset/kaggle_house/test.csv')
# print(train_data.shape, test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 先把数值全的特征标准化一下
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features = all_features.fillna(0)

# 离散数值转成指示特征, dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape) # (2919, 354)

#tensor获取数值
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

loss = nn.MSELoss()
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()

# def train(net, train_features,train_lables, test_features, test_lables, num_epochs, learning_rate, weight_decay, batch_sizes, loss):
    # train_loss, test_loss =

