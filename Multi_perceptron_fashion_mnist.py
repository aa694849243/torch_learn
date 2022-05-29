# -*- coding: utf-8 -*-
# author： caoji
# datetime： 2022-03-28 20:42 
# ide： PyCharm
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils import data as Data
from torchvision import transforms, datasets

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=True, transform=trans, download=True)
mnist_test = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=False, transform=trans, download=True)
NUM_INPUT = 784
NUM_OUTPUT = 10
NUM_HIDDENS = 512
BATCH_SIZE = 256
train_iter = Data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_iter = Data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class Flattenlayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


net = nn.Sequential(Flattenlayer(), nn.Linear(NUM_INPUT, NUM_HIDDENS), nn.ReLU(), nn.Linear(NUM_HIDDENS, NUM_OUTPUT))
loss = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.to(device)
for params in net.parameters():
    nn.init.normal_(params, mean=0, std=0.01)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
NUM_EPOCHS = 10
loss_history = []
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

acc = 0
for batch_imgs, batch_labels in test_iter:
    batch_imgs = batch_imgs.to(device)
    batch_labels = batch_labels.to(device)
    out = net(batch_imgs)
    out = out.argmax(axis=1)
    acc += (out == batch_labels).sum().item()
acc = acc / len(mnist_test)
print(acc)
