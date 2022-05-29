# -*- coding: utf-8 -*-
# author： caoji
# datetime： 2022-04-25 0:26 
# ide： PyCharm
import torch
import torch.nn as nn


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


net = torch.nn.Sequential(Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2, stride=2),
                          nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
# %%
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape :\t', X.shape)
# %%
from torch.utils import data as Data
from torchvision import transforms, datasets

mnist_train = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=False, transform=transforms.ToTensor(), download=True)
BATCH_SIZE = 256
train_iter = Data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_iter = Data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# %%
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = [0, 0]
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            metric[0] += torch.sum(torch.argmax(net(X), dim=1) == y).item()
            metric[1] += y.shape[0]
        return metric[0] / metric[1]


# %%
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        metric = [0, 0, 0]
        net.train()
        for i,(X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X=X.to(device)
            y=y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric[0] += l * X.shape[0]
            metric[1] += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
            metric[2] += y.shape[0]
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print('epoch %d, batch %d, loss %.4f, train acc %.3f' % (epoch + 1, i + 1, train_l, train_acc))
        n += y.shape[0]
    test_acc = evaluate_accuracy_gpu(net, test_iter)
    print(f'loss {train_l:.3f},train acc {train_acc:.3f},test acc {test_acc:.3f}')


# %%
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, 'cuda')
