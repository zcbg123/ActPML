# 这里是SAC的编辑
# 开发时间：2024/10/26 11:00
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
class LGCE(nn.Module):
    def __init__(self, channels, r=4):
        super(LGCE, self).__init__()
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


