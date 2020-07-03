import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
torch.set_default_tensor_type(torch.DoubleTensor)
'''load letter-recognition dataset'''
dataset = pd.read_csv(r'C:\Users\87408\Desktop\数据集\Letter\数据集-未划分训练集和测试集\letter-recognition.csv')
data = np.array(dataset.T)
# 将数据集划分为训练集和测试集
X_train = data[1:, :16000].T  # [16000, 16]
X_train_label = data[0:1, :16000].T  # [16000, 1]
for i in range(len(X_train_label)):
    X_train_label[i, :] = float(ord(str(X_train_label[i, :])[2]) - ord('A'))
X_train = torch.from_numpy(X_train.astype(float))
X_train_label = torch.from_numpy(X_train_label.astype(float))
X_test = data[1:, 16000:].T  # [3999, 16]
X_test_label = data[0:1, 16000:].T  # [3999, 1]
for i in range(len(X_test_label)):
    X_test_label[i, :] = float(ord(str(X_test_label[i, :])[2]) - ord('A'))
X_test = torch.from_numpy(X_test.astype(float))
X_test_label = torch.from_numpy(X_test_label.astype(float))
train_data = Data.TensorDataset(X_train, X_train_label)
test_data = Data.TensorDataset(X_test, X_test_label)
# torch处理数据
batch_size = 500
train_loader = Data.DataLoader(
    dataset=train_data,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=batch_size,      # 每块的大小
    shuffle=True               # 要不要打乱数据 (打乱比较好)
)
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=True
)


# 定义画图函数
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.title('train loss')
    plt.show()


# 将标签转为one-hot编码：
def one_hot(label, depth=26):
    out = torch.zeros(label.size(0), depth)
    for k in range(label.size(0)):
        out[k, int(label[k, :])] = 1
    return out


# 定义神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(16, 400)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 26)

    def forward(self, t):
        x1 = F.relu(self.fc1(t))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = self.fc4(x3)
        return x4
# 训练过程


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.1)
train_loss = []

for epoch in range(200):

    for batch_idx, (x, y) in enumerate(train_loader):
        # x: [b, 1, 28, 28], y:[512]
        # [b, 1, 28, 28] => [b, feature]
        # =>[b,10]
        out = net(x)
        y_onehot = one_hot(y)
        # loss = MSE(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())
plot_curve(train_loss)
# 测试过程
total_correct = 0
for x, y in test_loader:

    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)
plt.savefig('test2.jpg')

