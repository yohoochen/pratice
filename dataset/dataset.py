import torch
import numpy as np
import torch.utils.data as Data

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10

def getdata():
    dataset = Data.TensorDataset(features, labels)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)
    return data_iter

if __name__ == "__main__":
    data_iter = getdata()
    print(data_iter)
    for i in data_iter:
        print(i)
        break