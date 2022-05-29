# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from tqdm import tqdm

# %%
# 看看label文件长啥样
labels_dataframe = pd.read_csv('./dataset/classify_leaves/train.csv')
labels_dataframe.head(5)
# %%
np.asarray(labels_dataframe.iloc[2:10, 0])
# %%
labels_dataframe.describe()
# %%
leaves_labels = sorted(labels_dataframe['label'].unique())
leaves_labels
# %%
cls_to_num = dict(zip(leaves_labels, range(len(leaves_labels))))
cls_to_num
# %%
num_to_cls = dict(zip(range(len(leaves_labels)), leaves_labels))
num_to_cls
# %%
len(labels_dataframe)


# %%
class LeavesData(Dataset):
    def __init__(self, csv_path, img_path, mode='train', valid_ration=0.2, resize_h=256, resize_w=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """
        self.resize_h, self.resize_w = resize_h, resize_w
        self.img_path = img_path
        self.mode = mode
        self.data_info = pd.read_csv(csv_path, header=None)
        self.data_len = len(self.data_info) - 1
        self.train_data_len = int(self.data_len * (1 - valid_ration))
        # print(self.data_len)
        if mode == 'train':
            self.train_img = np.asarray(self.data_info.iloc[1:self.train_data_len, 0])
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_data_len, 1])
            self.img_arr = self.train_img
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_img = np.asarray(self.data_info.iloc[self.train_data_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_data_len:, 1])
            self.img_arr = self.valid_img
            self.label_arr = self.valid_label
        else:
            self.test_img = np.asarray(self.data_info.iloc[:, 0])
            self.img_arr = self.test_img
        self.real_len = len(self.img_arr)
        print(f'Finished reading the {mode} set of Leaves Dataset ({self.real_len} samples found)')

    def __getitem__(self, index):
        img_name = self.img_arr[index]
        img_as_img = Image.open(os.path.join(self.img_path, img_name))
        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')
        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(0.8,1.2)),#随机裁剪
                # transforms.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),#随机调整图片的亮度、对比度、饱和度、色调
                # transforms.Resize((224,224)),#缩放图片
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ToTensor(), ]  # 将图片转换成Tensor
            )
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 缩放图片
                transforms.ToTensor(), ]  # 将图片转换成Tensor
            )
        img_as_tensor = transform(img_as_img)
        if self.mode == 'test':
            return img_as_tensor
        else:
            label = self.label_arr[index]
            num_label = cls_to_num[label]
            return img_as_tensor, num_label

    def __len__(self):
        return self.real_len


# %%
train_path = 'dataset/classify_leaves/train.csv'
test_path = 'dataset/classify_leaves/test.csv'
img_path = 'dataset/classify_leaves/'
train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path, mode='valid')
test_dataset = LeavesData(test_path, img_path, mode='test')
print(train_dataset)
print(val_dataset)
print(test_dataset)
# %%
train_iter = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
val_iter = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
test_iter = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)


# %%
# 是否要冻住一些层
def set_parameter_requires_grad(model, feature_extracting):
    # if feature_extracting:
    #     model=model
    #     for i,param in enumerate(model.children()):
    #         if i==8:
    #             break
    #         param.requires_grad = False
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


# resnet34
def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Sequential(
    #         nn.Linear(num_ftrs, 512),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(.3),
    #         nn.Linear(512, len(num_to_class))
    #     )
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft


# resnext50模型
def resnext_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft


if __name__ == '__main__':
    # %%
    # 超参数
    learning_rate = 3e-4
    weight_decay = 1e-3
    num_epoch = 50
    model_path = 'dataset/leaf_classify/pre_resnext_model.ckpt'
    # %%
    # Initialize a model, and put it on the device specified.
    device = 'cuda'
    # model = res_model(176)
    model = resnext_model(176)
    model = model.to(device)
    model.device = device
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()
    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 余弦退火，last_epoch表示上一次训练的最后一个epoch,-1表示当前轮从0开始
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
    # The number of training epochs.
    n_epochs = num_epoch

    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_acc = []
        i = 0
        # Iterate the training set by batches.
        for batch in tqdm(train_iter):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i % 500 == 0):
                print("learning_rate:", scheduler.get_last_lr()[0])
            i += 1
            print((logits.argmax(dim=1) == labels))
            acc = (logits.argmax(dim=1) == labels).float().mean()
            train_loss.append(loss.item())
            train_acc.append(acc)
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        for batch in tqdm(val_iter):
            imgs, labels = batch
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #     # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))
