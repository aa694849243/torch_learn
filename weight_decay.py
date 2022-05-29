# -*- coding: utf-8 -*-
# author： caoji
# datetime： 2022-03-30 23:07 
# ide： PyCharm
# 权重衰减
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NUM_OUTPUT = 1
NUM_EPOCHS = 20
NUM_INPUT = 10
BATCH_SIZE = 5
device = 'cuda'
n_train, n_test = 20, 100
true_w, true_b = torch.ones((NUM_INPUT, 1)) * 0.01, 0.05


def get_data(sample_nums):
    X = torch.rand([sample_nums, 10])
    Y = X @ true_w + true_b + torch.normal(0, 1e-4, size=(sample_nums, 1))
    src, trg = X, Y
    return src, trg


src, trg = get_data(100)
data_train = TensorDataset(src, trg)
data_loader = DataLoader(data_train, batch_size=5, shuffle=False)
src2, trg2 = get_data(200)
data_test = TensorDataset(src2, trg2)
test_loader = DataLoader(data_test, batch_size=5, shuffle=False)


class Flattenlayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def train_concise(wd):
    net = nn.Sequential(Flattenlayer(), nn.Linear(NUM_INPUT, 1))
    for params in net.parameters():
        nn.init.normal_(params, mean=0, std=0.01)
    loss_history = []
    net.to('cuda')
    optimizer = torch.optim.SGD([{'params': net[1].weight, 'weight_decay': wd}, {'params': net[1].bias}], lr=0.01)
    loss = nn.MSELoss()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for i_batch, (X, Y) in enumerate(data_loader):
            X = X.to(device)
            Y = Y.to(device)
            y_hat = net(X)
            step_loss = loss(y_hat, Y)
            epoch_loss += step_loss

            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()
        loss_history.append(epoch_loss.item())
        print(f'{epoch=};{epoch_loss=}')
    test_loss = 0
    for i_batch, (X, Y) in enumerate(test_loader):
        X=X.to(device)
        Y=Y.to(device)
        y_hat = net(X)
        test_loss += loss(y_hat, Y)
    print(test_loss)


if __name__ == '__main__':
    train_concise(0)
    train_concise(0.1)
    train_concise(1)