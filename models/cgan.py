'''
 FILENAME:      cgan.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.16/23:36

 CONTACT:       duanxianshi@gmail.com
'''

from turtle import forward
import torch
import torch.nn as nn

import config


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, activation_func, normalize):
        super(FCBlock, self).__init__()

        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if normalize else None
        self.af = activation_func

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        if self.bn:
            y = self.bn(y)
        return self.af(y)

# class Generator(nn.Module):
#     def __init__(self, blocks, normalize):
#         super(Generator, self).__init__()
#         self.net = nn.Sequential(*[
#             FCBlock(b['in_features'], b['out_features'], b['activation_func'], normalize)
#             for b in blocks
#         ])

#     def forward(self, x, y):
#         z = torch.cat([x, y], dim=1)
#         z = self.net(z)
#         return z

class Generator(nn.Module):
    def __init__(self, blocks, normalize):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(110, 256)
        self.bn1 = nn.BatchNorm1d(256) if normalize else None
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512) if normalize else None
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024) if normalize else None
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024) if normalize else None
        self.relu4 = nn.ReLU(inplace=True)

        self.fc5 = nn.Linear(1024, 784)
        self.bn5 = nn.BatchNorm1d(784) if normalize else None
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)

        z = self.fc1(z)
        if self.bn1:
            z = self.bn1(z)
        z = self.relu1(z)

        z = self.fc2(z)
        if self.bn2:
            z = self.bn2(z)
        z = self.relu2(z)

        z = self.fc3(z)
        if self.bn3:
            z = self.bn3(z)
        z = self.relu3(z)

        z = self.fc4(z)
        if self.bn4:
            z = self.bn4(z)
        z = self.relu4(z)

        z = self.fc5(z)
        if self.bn5:
            z = self.bn5(z)
        z = self.tanh(z)

        return z


# class Discriminator(nn.Module):
#     def __init__(self, blocks, normalize, drop_ratio):
#         super(Discriminator, self).__init__()
#         self.dropout = nn.Dropout(drop_ratio)
#         self.net = nn.Sequential(*[
#             FCBlock(b['in_features'], b['out_features'], b['activation_func'], normalize)
#             for b in blocks
#         ])

#     def forward(self, x, y):
#         x_ = self.dropout(x)
#         z = torch.cat([x_, y], dim=1)
#         z = self.net(z)
#         return z

class Discriminator(nn.Module):
    def __init__(self, blocks, normalize, drop_ratio):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(drop_ratio)

        self.fc1 = nn.Linear(794, 512)
        self.bn1 = nn.BatchNorm1d(512) if normalize else None
        # self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256) if normalize else None
        # self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(1024) if normalize else None
        # self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(1024) if normalize else None
        # self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.fc5 = nn.Linear(64, 1)
        self.bn5 = nn.BatchNorm1d(1) if normalize else None

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.af = nn.Sigmoid()

    def forward(self, x, y):
        x_ = self.dropout(x)
        z = torch.cat([x_, y], dim=1)

        z = self.fc1(z)
        if self.bn1:
            z = self.bn1(z)
        z = self.relu(z)

        z = self.fc2(z)
        if self.bn2:
            z = self.bn2(z)
        z = self.relu(z)

        z = self.fc3(z)
        if self.bn3:
            z = self.bn3(z)
        z = self.relu(z)

        z = self.fc4(z)
        if self.bn4:
            z = self.bn4(z)
        z = self.relu(z)

        z = self.fc5(z)
        if self.bn5:
            z = self.bn5(z)
        z = self.af(z)
        return z