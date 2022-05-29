# -*- coding: utf-8 -*-
# author： caoji
# datetime： 2022-03-30 23:08 
# ide： PyCharm
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils import data as Data
from torchvision import transforms, datasets

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BATCH_SIZE = 256
trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=True, transform=trans, download=True)
mnist_test = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=False, transform=trans, download=True)
train_iter = Data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_iter = Data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.randn(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


dropout1, dropout2 = 0.2, 0.3
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2),
                    nn.Linear(256, 10))
device = 'cuda'
net.to(device)

# 参数初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)


net.apply(init_weights)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss = nn.CrossEntropyLoss()
NUM_EPOCHS = 10
loss_history = []
net.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch_imgs, y in train_iter:
        batch_imgs = batch_imgs.to(device)
        y = y.to(device)
        y_hat = net(batch_imgs)
        step_loss = loss(y_hat, y)
        epoch_loss += step_loss

        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()
    loss_history.append(epoch_loss)
    print(f'{epoch=};{epoch_loss=}')

fig, ax = plt.subplots()
ax.plot(np.arange(NUM_EPOCHS) + 1, loss_history)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()
net.eval()
acc = 0
for batch_imgs, batch_labels in test_iter:
    batch_imgs = batch_imgs.to(device)
    batch_labels = batch_labels.to(device)
    out = net(batch_imgs)
    out = out.argmax(axis=1)
    acc += (out == batch_labels).sum().item()
acc = acc / len(mnist_test)
print(acc)
# if __name__ == '__main__':
#     X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
#     print(X)
#     print(dropout_layer(X, 0.5))
