# -*- coding: utf-8 -*-
# author： caoji
# datetime： 2022-05-07 23:53 
# ide： PyCharm
import torch
import torch.nn as nn
from torch.utils import data as Data
from torchvision import transforms, datasets
#%%
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # momentum 一般选0.9或0.1
    if not torch.is_grad_enabled():  #做推理的时候用全局的均值和方差，因为可能就一个样本
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)  #2维为全连接层，4维为卷积层
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)  #每一个通道的均值
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean  #用当前批量的均值更新全局的均值
        moving_var = momentum * moving_var + (1 - momentum) * var
    Y = gamma * X_hat + beta  #更新gamma和beta
    return Y, moving_mean.data, moving_var.data
#%%
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y
#%%
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
                    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
                    nn.Linear(84, 10))
net2 = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
#%%
lr, num_epochs, batch_size = 1., 10, 256
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
mnist_train = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=True, transform=transform, download=True)
mnist_test = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=False, transform=transform, download=True)
train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)


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


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

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
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
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
# train_ch6(net, train_iter, test_iter, num_epochs, lr, 'cuda')
X=torch.rand(size=(1,1,224,224))
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)