import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
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
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


BATCH_SIZE = 256
EPOCH = 100
train_dataset = MNIST(root='./train', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='./test', train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = CNN()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

f = open('logs.txt', 'w')

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
