import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, X):
        Y = self.conv1(X)
        Y = F.relu(Y)
        Y = self.conv2(Y)
        Y = X + F.relu(Y)

        return Y

class Network(nn.Module):
    def __init__(self, in_channels, n_res_blocks=19, n_filters=256,
                 compress_filters=100):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, n_filters, 3, padding=1) # (256,8,8)
        self.tower = nn.Sequential(*[
            ResidualBlock(n_filters, n_filters) for _ in range(n_res_blocks)
        ]) # (256, 8, 8)
        self.compress = nn.Conv2d(n_filters, compress_filters, 1)

        self.value_head = nn.Linear(compress_filters * 64, 1)
        self.policy_head = nn.Linear(compress_filters * 64, 8 * 8 * 73)

    def forward(self, X):
        Y = self.conv(X)
        Y = F.relu(Y)
        Y = self.tower(Y)
        compressed = self.compress(Y)
        compressed = F.relu(compressed)
        flattened = compressed.flatten()

        value = self.value_head(flattened)
        policy = self.policy_head(flattened).reshape(-1, 8, 8, 73)

        return value, policy
