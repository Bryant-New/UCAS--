import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

'''设置基本信息'''
'''PyTorch中数据按batch批量输入,设置batch和学习率以及训练次数'''
batch_size = 200
learning_rate = 0.01
epochs = 10
# 指定训练集和测试集的数据加载器：
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)
# torch.utils.data.DataLoader数据加载器的接口，第一个参数传数据集，第二个参数batch_size表示每个batch加载多少个样本
# 第三个参数shuffle设置为True表示为每个epoch重新打乱数据。
# datasets.MNIST()方法，第一个参数’…/data’表示 processed/training.pt 和 processed/test.pt的主目录
# train设置为False表示测试集，True为训练集，download设置为True表示从互联网上下载数据集
# transform表示一个函数，输入为target，输出对其的转换
# transforms.ToTensor()将取值范围[0,255]的image或者shape为(H,W,C)转化为形状[C,H,W],取值范围[0,1],将普通图片数据转为tensor数据
# transforms.Normalize((0.1307,), (0.3081,))表示给定均值-标准化-数据减去均值，再除以标准差，数据均值为0方差为1的分布，标准化后数据效果更好
# 分布不均匀的数据在经过激活函数求导之后，得到的结果接近于0-梯度消失，均值是0.1307，标准差是0.3081

# 初始化线性层，会初始化一个全0或全1亦或是随机初始值的特定维度的张量。
w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)
# w-权值，b-bias向量，randn 表示返回一个张量，从标准正态分布(均值为0，方差为 1，即高斯白噪声)中抽取一组随机数
# 第一个线性层，输入784维，输出200维，输入像素28*28的数字图片
# 隐含层，隐含层的输入输出都是200维，激活函数变换
# 输出层，输出一个10维特征

# 初始化前向传播过程
# 使用ReLU作为激活函数，避免梯度消失现象
def forward(x):
    x = x@w1.t()+b1
    x = F.relu(x)
    x = x@w2.t()+b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x

# 定义优化器和优化的标准
optimizer = optim.SGD([w1, b1, w2, b2,w3, b3])
criteon = nn.CrossEntropyLoss()
# SGD优化算法，传入模型参数和学习率，就可计算函数的交叉熵了

# 模型训练
for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)

        logits = forward(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. *batch_idx / len(train_loader),loss.item()))
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28)
        logits = forward(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct +=pred.eq(target.data).sum()

    test_loss /=len(test_loader.dataset)
    print('\nTest set: Average loss:{:.4f},Accuracy:{}/{}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
# data.view(-1,28 * 28)返回一个相同数据但大小不同的tensor，-1计算该维度的大小
# 交叉熵函数-LogSoftMax、NLLLoss，第一个logits表示每个类别的得分,target是大小为n的1—D tensor-张量，tensor就是深度学习框架的基本数据类型
# 会初始化一个全0或全1亦或是随机初始值的特定维度的张量。整个函数里面包含了softmax函数，优化完损失函数之后就不需要再进行softmax操作了
# zero_grad()表示清空所有被优化过的Variable的梯度，loss.backward()反向传播更新权重，optimizer.step()进行单次的优化操作。

# 使用何凯明方法提高模型准确率,kaiming初始化
torch.nn.init.kaiming_normal(w1)
torch.nn.init.kaiming_normal(w2)
torch.nn.init.kaiming_normal(w3)



