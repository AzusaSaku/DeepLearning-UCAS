## 手写数字识别实验

---

### 1. 实验目的
#### 1.1 任务概述

本次实验旨在学习掌握卷积神经网络的基本原理，以及使用PyTorch构建CNN网络的基本操作，实验数据集为MNIST手写数字数据集，该数据集大小（1，28，28），为单通道的灰度图像。

实验结果：
>+ 学习搭建人工神经网络开发环境（√）
>+ 学习构建规范的卷积神经网络组织结构（√）
>+ 在MNIST手写数字数据集上进行训练和评估，实现测试集准确率达到98%及以上（√）

#### 1.2 实验环境

实验采用的PyTorch版本如下图所示：

![image](image/1.png)

---

### 2. 实验过程

#### 2.1 数据获取

下载实验所需的MNIST数据集，分别存放在/train和/test文件夹。其中train参数为真时表示训练集，为假时表示测试集，download参数在第一次运行时将数据下载到指定路径位置。
``````python
train_dataset = MNIST(root='./train', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='./test', train=False, download=True, transform=ToTensor())
``````
使用torch.utils.data中的DataLoader方法包装数据，将数据分为训练集和测试集。
``````python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
``````

#### 2.2 参数选择

首先采用早期的网络结构，发现早期MNIST数据集和当前的不同，于是选择使用指导书中介绍的网络结构，针对该结构选择三种学习率（1e-1、1e-3、1e-4）分别进行实验，实验结果表明学习率为1e-4时效果最好，训练时间适中，最终选择该方案。

实验结束后，针对相同数据集，采用三层卷积的CNN和MLP分别进行实验，对比实验结果如下表所示：

| 网络结构 | 学习率	| 效果 | 备注 |
| :----: | :----: | :----: | :----: |
| LeNet-5 |  |  | 数据结构变化 |
| 指导书 | 1e-1	| 0.95 | 可能未收敛 |
|  | 1e-3 | 0.98 |  |
|  | 1e-4 | 0.99 | 最终选用 |
| 三层卷积 | 1e-1 | 未收敛 | padding=0 |
|  | 1e-3 | 0.98 |  |
| 多层感知机 | 1e-3 | 0.91 |  |

#### 2.3 代码分析

构造CNN网络的第一个卷积层，由于MNIST数据集为(（1，28，28）)的单通道图像，于是采用大小为（16，5，5）、步长为1的卷积核，同时填充padding=2以确保输出与输入尺寸相同，此时输出图像大小为（16，28，28）。
``````python
nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=5,
    stride=1,
    padding=2)
``````
采用ReLU作为激活函数。
``````python
nn.ReLU()
``````
池化层在2*2的空间进行下采样，此时输出图像大小为（16，14，14）。
``````python
nn.MaxPool2d(kernel_size=2)
``````
与上述卷积层类似，建立第二个卷积层，输入数据大小为（16，14，14），输出数据大小为（32，7，7）。
``````python
self.conv2 = nn.Sequential(
    nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=5,
        stride=1,
        padding=2
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)
``````
构造全连接层，输入数据大小为（32，7，7），输出为10个类。
``````python
self.out = nn.Linear(32 * 7 * 7, 10)
``````
定义前向传播函数，实验数据经过conv1和conv2卷积层后，由view方法转化为特定个张量。
``````python
def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    output = self.out(x)
    return output
``````

#### 2.4 模型训练

选用Adam优化器和交叉熵损失，对不同学习率进行多次实验，当lr=1e-4时效果最好。
``````python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()
``````
EPOCH=100，BATCH_SIZE=256，开始训练模型，通过反向传播计算梯度，注意每次计算前需要清除先前的梯度。
``````python
for epoch in range(EPOCH):
    for idx, (train_x, train_label) in enumerate(train_loader):
        output = model(train_x)
        loss = loss_func(output, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 50 == 0:
            correct = total = 0
            with torch.no_grad():
                for _, (b_x, b_y) in enumerate(test_loader):
                    output = model(b_x)
                    predict_y = torch.max(output, 1)[1].detach().numpy()
                    correct += float((predict_y == b_y.detach().numpy()).astype(int).sum())
                    total += float(b_y.size(0))
                accuracy = correct / total
                f.write('Epoch: %d |train loss: %.4f |test accuracy: %.2f\n' % (epoch, loss.data.numpy(), accuracy))
                print('Epoch: %d |train loss: %.4f |test accuracy: %.2f' % (epoch, loss.data.numpy(), accuracy))

torch.save(model.state_dict(), 'cnn.pth')
``````
训练完成后，将模型保存为cnn.pth文件，训练过程保存于logs.txt文件中。

### 3. 实验分析

通过本次实验，基本掌握了人工神经网络的训练和调参过程，也为接下来的几次实验打好了基础。本次实验过程中主要选用了三种模型（指导书、三层卷积、MLP）在不同学习率下进行实验，有如下几个现象及结论：

>+ 学习率较低时，模型可能会无法收敛
>+ 使用更复杂的模型（三层卷积），训练速度降低，但准确率没有显著影响
>+ 针对上一条，原因可能是实验数据过于简单，但更复杂的模型泛化能力可能受到影响
>+ 多层感知机虽然可解释性高，结构优雅，但在此实验中效果远不如CNN网络

权重文件和训练日志见文件夹内。