import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)

class Architecture(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()

    def add(self, module):
        self.blocks.append(module)
        return self

    def residual(self, *layers):
        self.blocks.append(ResidualBlock(*layers))
        return self

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x