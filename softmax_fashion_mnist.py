# -*- coding: utf-8 -*-
# author： caoji
# datetime： 2022-03-26 23:41 
# ide： PyCharm
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as Data
from torchvision import transforms, datasets

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mnist_train = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.FashionMNIST(root='dataset/Fashion_Minist', train=False, download=True, transform=transforms.ToTensor())
print(len(mnist_train), len(mnist_test))
print(mnist_train[0][0].size())


# plt.figure(1)
# plt.imshow(mnist_train[0][0].view(28,28).numpy())
# plt.show()
def get_type(labels):
    types = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
             'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [types[int(i)] for i in labels]


def display_images(imgs, labels):
    fig, ax = plt.subplots(1, len(imgs), figsize=(18, 20))
    types = get_type(labels)
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i])
        ax[i].set_title(types[i])
    plt.show()


# print(get_type([0,1,2,3]))
imgs = [mnist_train[i][0].view(28, 28) for i in range(10)]
labels = [mnist_train[i][1] for i in range(10)]
display_images(imgs, labels)

BATCH_SIZE = 256
train_iter = Data.DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True)
test_iter = Data.DataLoader(dataset=mnist_test, batch_size=BATCH_SIZE, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(28 * 28, 10)

    def forward(self, img):
        out = self.layer(img)
        return out


net1 = Softmax()
net1.to(device)
print(net1)

torch.nn.init.normal_(net1.layer.weight, mean=0.0, std=0.01)
torch.nn.init.constant_(net1.layer.bias, val=0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net1.parameters(), lr=0.01)

EPOCH_NUM = 20
loss_history = []
for epoch in range(EPOCH_NUM):
    epoch_loss = 0
    for batch_imgs, batch_labels in train_iter:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = net1(batch_imgs.view(batch_imgs.shape[0], -1))

        step_loss = loss(out, batch_labels)
        epoch_loss += step_loss

        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()
    loss_history.append(epoch_loss)
    print(f'{epoch=};{epoch_loss=}')

fig, ax = plt.subplots()
ax.plot(np.arange(EPOCH_NUM) + 1, loss_history)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()

acc = 0
for batch_imgs, batch_labels in test_iter:
    batch_imgs = batch_imgs.to(device)
    batch_labels = batch_labels.to(device)
    out = net1(batch_imgs.view(batch_imgs.shape[0], -1))
    out = out.argmax(axis=1)
    acc += (out == batch_labels).sum().item()
acc=acc/len(mnist_test)
print(acc)
