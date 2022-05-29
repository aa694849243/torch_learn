# -*- coding: utf-8 -*-
# author： caoji
# datetime： 2022-05-28 5:30 
# ide： PyCharm
import os
import time
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# %%
class TrainData(Dataset):
    def __init__(self, csv_path, img_path, transform):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """
        self.img_path = img_path
        self.data_info = pd.read_csv(csv_path)
        self.data_len = len(self.data_info)
        self.train_img = np.asarray(self.data_info.iloc[: self.data_len, 0])
        self.train_label = np.asarray(self.data_info.iloc[: self.data_len, 1])
        self.transform = transform
        self.img_arr = self.train_img
        self.label_arr = self.train_label
        # print(self.img_arr)
        # print('*' * 30)
        # print(len(self.img_arr))

    def __getitem__(self, index):
        # print(index)
        img_name = self.img_arr[index]
        img_as_img = Image.open(os.path.join(self.img_path, f"{img_name}.png"))
        img_as_tensor = self.transform(img_as_img)
        label = self.label_arr[index]
        num_label = cls_to_num[label]
        return img_as_tensor, num_label

    def __len__(self):
        return self.data_len




# %%


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


# ResNext模型
def resnet_model(num_classes, feature_extracting=False):
    model = models.resnet18(pretrained=False)
    set_parameter_requires_grad(model, feature_extracting)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model


# %%
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


transform_train = torchvision.transforms.Compose(
    [
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64到1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(
            32, scale=(0.64, 1.0), ratio=(1.0, 1.0)
        ),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        ),
    ]
)
transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        ),
    ]
)
# %%
train_path = "dataset/cifar-10/trainLabels.csv"
train_img_path = "dataset/cifar-10/train/"
labels_dataframe = pd.read_csv("./dataset/cifar-10/trainLabels.csv")
leaves_labels = sorted(list(set(labels_dataframe["label"])))
n_classes = len(leaves_labels)
# %%
cls_to_num = dict(zip(leaves_labels, range(len(leaves_labels))))
num_to_cls = dict(zip(range(len(leaves_labels)), leaves_labels))
train_set = TrainData(
    csv_path=train_path, img_path=train_img_path, transform=transform_train
)
ids = np.random.permutation(range(len(labels_dataframe)))
split = int(len(labels_dataframe) * 0.9)
train_ids, valid_ids = ids[:split], ids[split:]
# print(max(train_ids))
# print(max(valid_ids))
train_subsampler = torch.utils.data.SubsetRandomSampler(np.array(train_ids))
valid_subsampler = torch.utils.data.SubsetRandomSampler(np.array(valid_ids))
if __name__ == '__main__':
    # freeze_support()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        sampler=train_subsampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        sampler=valid_subsampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    # for a in tqdm(range(10)):
    #     print(a)
    # for batch in tqdm(train_loader):
    #     print(batch[0].shape)
    #     print(batch[1])
    batch_size = 128
    device, num_epochs, lr, wd = 'cuda', 2, 2e-4, 5e-4
    lr_period, lr_decay = 4, 0.9
    loss_function = nn.CrossEntropyLoss()
    model = resnet_model(10)
    model.apply(init_weights)
    model = model.to(device)
    model.device = device
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # 线性衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    best_acc = 0.
    model_path = './models/tiny_cifar.pth'
    # print(max(train_ids))
    # print(max(valid_ids))
    for epoch in range(num_epochs):
        # time.sleep(0.5)
        model.train()
        print(f'Starting epoch {epoch + 1}')
        print('*' * 30)
        train_loss = []
        train_acc = []
        cnt=1
        for batch in tqdm(train_loader):
            print(f'batch {cnt}')
            cnt+=1
            time.sleep(0.005)
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_function(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss.append(loss.item())
        print("第%d个epoch的学习率：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
        scheduler.step()
        train_avg_loss = np.sum(train_loss) / len(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_avg_loss:.5f}")
        print('Training process has finished. Saving trained model.')
        print('Starting validation')
        model.eval()
        valid_loss = []
        valid_acc = []
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                time.sleep(.005)
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = loss_function(logits, labels)
                valid_loss.append(loss.cpu().item())
                valid_acc.append((logits.argmax(dim=-1) == labels).float().mean().cpu())
            valid_avg_loss = np.sum(valid_loss) / len(valid_loss)
            valid_avg_acc = np.sum(valid_acc) / len(valid_acc)
            print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_avg_loss:.5f}, acc = {valid_avg_acc:.5f}")
            if valid_avg_acc > best_acc:
                best_acc = valid_avg_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc))
