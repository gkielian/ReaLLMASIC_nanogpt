import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, ndim):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / rms * self.gain


class PowerNorm(nn.Module):
    """Power Normalization"""

    def __init__(self, ndim, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.gain = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.pow(var + 1e-5, self.alpha)
        return normalized * self.gain


class GroupNorm(nn.GroupNorm):
    """Group Normalization with gain parameter"""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__(num_groups, num_channels, eps)
        self.gain = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        normalized = super().forward(x)
        return normalized * self.gain


class InstanceNorm(nn.InstanceNorm2d):
    """Instance Normalization with gain parameter"""

    def __init__(self, num_features, eps=1e-5):
        super().__init__(num_features, eps)
        self.gain = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        normalized = super().forward(x)
        return normalized * self.gain


class BatchNorm(nn.BatchNorm1d):
    """BatchNorm1d with gain parameter"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(num_features, eps, momentum)
        self.gain = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        # Reshape input for BatchNorm1d
        x = x.transpose(1, 2)
        normalized = super().forward(x)
        # Reshape back to original shape
        x = normalized.transpose(1, 2)
        return x * self.gain
