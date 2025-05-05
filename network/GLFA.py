# 这里是SAC的编辑
# 开发时间：2024/8/21 11:19
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
class LGAE4(nn.Module):
    def __init__(self, channels=512, r=4):
        super(LGAE4, self).__init__()
        inner_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lattn_feats = self.local_att(x)
        gattn_feats = self.global_att(x)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return x * w


class LGAA4(LGAE4):
    def forward(self, x1, x2):
        addition_feats = x1 + x2
        lattn_feats = self.local_att(addition_feats)
        gattn_feats = self.global_att(addition_feats)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return 2 * x1 * w + 2 * x2 * (1 - w)



class LGAE3(nn.Module):
    def __init__(self, channels=512, r=4):
        super(LGAE3, self).__init__()
        inner_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lattn_feats = self.local_att(x)
        gattn_feats = self.global_att(x)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return x * w


class LGAA3(LGAE3):
    def forward(self, x1, x2):
        addition_feats = x1 + x2
        lattn_feats = self.local_att(addition_feats)
        gattn_feats = self.global_att(addition_feats)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return 2 * x1 * w + 2 * x2 * (1 - w)



class LGAE2(nn.Module):
    def __init__(self, channels=256, r=4):
        super(LGAE2, self).__init__()
        inner_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lattn_feats = self.local_att(x)
        gattn_feats = self.global_att(x)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return x * w


class LGAA2(LGAE2):
    def forward(self, x1, x2):
        addition_feats = x1 + x2
        lattn_feats = self.local_att(addition_feats)
        gattn_feats = self.global_att(addition_feats)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return 2 * x1 * w + 2 * x2 * (1 - w)


class LGAE1(nn.Module):
    def __init__(self, channels=128, r=4):
        super(LGAE1, self).__init__()
        inner_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lattn_feats = self.local_att(x)
        gattn_feats = self.global_att(x)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return x * w


class LGAA1(LGAE1):
    def forward(self, x1, x2):
        addition_feats = x1 + x2
        lattn_feats = self.local_att(addition_feats)
        gattn_feats = self.global_att(addition_feats)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return 2 * x1 * w + 2 * x2 * (1 - w)