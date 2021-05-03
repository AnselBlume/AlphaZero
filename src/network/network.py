import torch
import torch.nn as nn
from torch.nn import functional as F
from network.encode_state import M, L

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
    def __init__(self, T, temp=2, n_res_blocks=19, n_filters=256,
                 compress_filters=100):
        super().__init__()

        self.T = T
        self.temp = temp

        in_channels = M * T + L
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
        flattened = compressed.reshape(X.shape[0], -1)

        value = torch.tanh(self.value_head(flattened))
        policy = F.log_softmax(self.policy_head(flattened) / self.temp, dim=1) \
                  .reshape(-1, 8, 8, 73)

        return value, policy
