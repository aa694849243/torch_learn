import os

import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# %%
torch.set_default_tensor_type('torch.FloatTensor')
print(torch.__version__)
# %%
# Load the data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
# %%
train_data.head()
# %%
print(train_data.shape, test_data.shape)
# %%
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
train_features = all_features.iloc[:train_data.shape[0], 1:-1]
train_labels = all_features.iloc[:train_data.shape[0], -1]
# %% md
### 数据处理
# %%
# 数值数据标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 填充均值0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# %%
all_features = pd.get_dummies(all_features, dummy_na=True)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to('cuda')
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to('cuda')
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32).to('cuda')
# %%
# loss
loss = nn.MSELoss()


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)


def get_net(feature_num):
    net = nn.Sequential(
        nn.Flatten(), nn.Linear(feature_num, 1)
    )
    net.apply(init_weights)
    return net


# %%
# 对数误差
def log_rmse(net, features, label):
    clipped_preds = torch.clamp(net(features), min=1.0, max=float('inf'))
    rmse = torch.sqrt(loss(clipped_preds.log(), label.log()))
    return rmse.item()


# %%
def train(net, train_features, train_label, test_features, test_label, batch_size, epoch, lr, wd):
    dataset = torch.utils.data.TensorDataset(train_features, train_label)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    net = net.float()  # 32位
    net = net.to('cuda')
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=wd)
    train_ls, test_ls = [], []
    for _ in range(epoch):
        for X, Y in train_iter:
            optimizer.zero_grad()
            train_pred = net(X.float())
            train_loss = loss(train_pred, Y.float())
            train_loss.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_label))
        if test_label is not None:
            test_ls.append(log_rmse(net, test_features, test_label))
    return train_ls, test_ls


# %% md
### K折交叉验证
# %%
def get_k_fold_data(k, i, X, Y):
    # 第i折作为测试集，其余作为训练集
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, Y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, Y_part = X[idx, :], Y[idx]
        if j == i:
            X_valid, Y_valid = X_part, Y_part
        elif X_train is None:
            X_train, Y_train = X_part, Y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            Y_train = torch.cat((Y_train, Y_part), dim=0)
    return X_train, Y_train, X_valid, Y_valid


# %%
def k_fold(k, X_train, Y_train, batch_size, epoch, lr, wd):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        X_train, Y_train, X_valid, Y_valid = get_k_fold_data(k, i, X_train, Y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, X_train, Y_train, X_valid, Y_valid, batch_size, epoch, lr, wd)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'{i + 1}折，训练误差：{train_ls[-1]}, 验证误差：{valid_ls[-1]}')
    return train_l_sum / k, valid_l_sum / k


# %%
k, epoch, lr, wd, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, batch_size, epoch, lr, wd)


# %%
def train_and_pred(train_features, train_labels, test_features, test_data, batch_size, epochs, lr, wd):
    net = get_net(train_features.shape[1]).to('cuda')
    train_ls, _ = train(net, train_features, train_labels, None, None, batch_size, epochs, lr, wd)
    print(f'训练误差：{train_ls[-1]}')
    plt.plot(train_ls, label='epoch')
    plt.yscale("log")
    plt.legend()
    plt.show()
    test_preds = net(test_features).detach().to('cpu').numpy()
    test_data['SalePrice'] = pd.Series(test_preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)
    print('finish')
    return test_preds


# %%
train_and_pred(train_features, train_labels, test_features, test_data, batch_size, epoch, lr, wd)
# %%
