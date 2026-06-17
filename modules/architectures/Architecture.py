import torch
from torch import nn


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, conv_layers, shortcut=None):
        super().__init__()
        self.conv = nn.Sequential(*conv_layers)
        self.shortcut = shortcut

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        return identity + self.conv(x)

class Architecture(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()

    def add(self, *modules):
        if len(modules) == 1:
            self.blocks.append(modules[0])
        else:
            self.blocks.append(nn.Sequential(*modules))
        return self

    def residual(self, *layers):
        self.blocks.append(ResidualBlock(*layers))
        return self

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x