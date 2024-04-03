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
        result = F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        # print(result.size())
        return result


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

class ManualBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.gain = nn.Parameter(torch.ones(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # Reshape input for BatchNorm1d
        x = x.transpose(1, 2)

        if True:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = (x * self.weight + self.bias)

        # Reshape back to original shape
        x = x.transpose(1, 2)

        return x * self.gain

class PowerNorm2(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha=0.9):
        super().__init__()
        self.num_features = num_features  # scalar, number of features/channels
        self.eps = eps  # scalar, small constant for numerical stability
        self.alpha = alpha  # scalar, exponential moving average factor
        self.gain = nn.Parameter(torch.ones(num_features))  # (C,), learnable gain parameter
        self.running_quad_mean = nn.Parameter(torch.ones(num_features), requires_grad=False)  # (C,), running quadratic mean

    def forward(self, x):
        # input x: (B, T, C), where B: batch size, T: sequence length, C: number of features/channels
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D input"
        B, T, C = x.shape  # scalars, get dimensions of input tensor

        # compute quadratic mean
        quad_mean = torch.mean(x ** 2, dim=(0, 1))  # (C,), compute quadratic mean along batch and sequence dimensions

        # normalize using quadratic mean
        x_normalized = x / (quad_mean.view(1, 1, C) + self.eps).sqrt()  # (B, T, C), normalize input by quadratic mean

        # update running quadratic mean
        self.running_quad_mean.data = self.alpha * self.running_quad_mean.data + (1 - self.alpha) * quad_mean.view_as(self.running_quad_mean.data)  # (C,), update running quadratic mean using exponential moving average

        # apply gain
        output = x_normalized * self.gain.view(1, 1, C)  # (B, T, C), apply learnable gain parameter

        return output  # (B, T, C), return normalized output

    def backward(self, grad_output):
        # input grad_output: (B, T, C), gradient of loss with respect to output
        assert grad_output.dim() == 3, f"Expected 3D grad_output, got {grad_output.dim()}D grad_output"
        B, T, C = grad_output.shape  # scalars, get dimensions of gradient tensor

        # compute intermediate gradient
        grad_x_hat = grad_output * self.gain.view(1, 1, C)  # (B, T, C), multiply gradient by gain parameter

        # compute approximate backpropagation
        x_hat = grad_output / (self.running_quad_mean.view(1, 1, C) + self.eps).sqrt()  # (B, T, C), normalize gradient by running quadratic mean
        quad_mean_grad = torch.sum(grad_x_hat * x_hat, dim=(0, 1), keepdim=True)  # (1, 1, C), compute gradient of quadratic mean
        grad_input = grad_x_hat - quad_mean_grad * x_hat / (2 * (B * T) * (self.running_quad_mean.view(1, 1, C) + self.eps).sqrt())  # (B, T, C), compute gradient with respect to input using approximate backpropagation

        return grad_input  # (B, T, C), return gradient with respect to input

class PowerNorm3(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gain = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.p = nn.Parameter(torch.ones(1))
        self.q = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # input shape: (B, T, C)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)

        # apply power normalization
        x = torch.sign(x) * torch.pow(torch.abs(x), self.p / self.q)

        # apply gain and bias
        if self.affine:
            x = x * self.gain.view(1, 1, -1) + self.bias.view(1, 1, -1)

        return x

class PowerNorm5(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha=0.9, p=2):
        super().__init__()
        self.num_features = num_features  # scalar, number of features/channels
        self.eps = eps  # scalar, small constant for numerical stability
        self.alpha = alpha  # scalar, exponential moving average factor
        self.p = p  # scalar, power for the power mean
        self.gain = nn.Parameter(torch.ones(num_features))  # (C,), learnable gain parameter
        self.running_power_mean = nn.Parameter(torch.ones(num_features), requires_grad=False)  # (C,), running power mean

    def forward(self, x):
        # input x: (B, T, C), where B: batch size, T: sequence length, C: number of features/channels
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D input"
        B, T, C = x.shape  # scalars, get dimensions of input tensor

        # compute power mean
        power_mean = torch.mean(torch.abs(x) ** self.p, dim=(0, 1))  # (C,), compute power mean across batch and sequence dimensions

        # normalize using power mean
        x_normalized = x / (power_mean.view(1, 1, C) + self.eps) ** (1 / self.p)  # (B, T, C), normalize input by power mean

        # update running power mean
        self.running_power_mean.data = self.alpha * self.running_power_mean.data + (1 - self.alpha) * power_mean  # (C,), update running power mean using exponential moving average

        # apply gain
        output = x_normalized * self.gain  # (B, T, C), apply learnable gain parameter

        return output  # (B, T, C), return normalized output

    def backward(self, grad_output):
        # input grad_output: (B, T, C), gradient of loss with respect to output
        assert grad_output.dim() == 3, f"Expected 3D grad_output, got {grad_output.dim()}D grad_output"
        B, T, C = grad_output.shape  # scalars, get dimensions of gradient tensor

        # compute intermediate gradient
        grad_x_hat = grad_output * self.gain.view(1, 1, C)  # (B, T, C), multiply gradient by gain parameter

        # compute approximate backpropagation
        x_hat = grad_output / (self.running_power_mean.view(1, 1, C) + self.eps) ** (1 / self.p)  # (B, T, C), normalize gradient by running power mean
        power_mean_grad = torch.sum(grad_x_hat * torch.abs(x_hat) ** (self.p - 1) * torch.sign(x_hat), dim=(0, 1), keepdim=True)  # (1, 1, C), compute gradient of power mean
        grad_input = grad_x_hat - power_mean_grad * x_hat / (self.p * (B * T) * (self.running_power_mean.view(1, 1, C) + self.eps) ** ((self.p - 1) / self.p))  # (B, T, C), compute gradient with respect to input using approximate backpropagation

        return grad_input  # (B, T, C), return gradient with respect to input

class PowerNorm5(nn.Module):
    def __init__(self, num_features, eps=1e-5, alpha=0.9):
        super().__init__()
        self.num_features = num_features  # scalar, number of features/channels
        self.eps = eps  # scalar, small constant for numerical stability
        self.alpha = alpha  # scalar, exponential moving average factor
        self.gain = nn.Parameter(torch.ones(num_features))  # (C,), learnable gain parameter
        self.running_quad_mean = nn.Parameter(torch.ones(num_features), requires_grad=False)  # (C,), running quadratic mean

    def forward(self, x):
        # input x: (B, T, C), where B: batch size, T: sequence length, C: number of features/channels
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D input"
        B, T, C = x.shape  # scalars, get dimensions of input tensor

        # compute quadratic mean
        quad_mean = torch.mean(x ** 2, dim=1)  # (B, C), compute quadratic mean along sequence dimension

        # normalize using quadratic mean
        x_normalized = x / (quad_mean.view(B, 1, C) + self.eps).sqrt()  # (B, T, C), normalize input by quadratic mean

        # update running quadratic mean
        self.running_quad_mean.data = self.alpha * self.running_quad_mean.data + (1 - self.alpha) * quad_mean.mean(dim=0)  # (C,), update running quadratic mean using exponential moving average

        # apply gain
        output = x_normalized * self.gain  # (B, T, C), apply learnable gain parameter

        return output  # (B, T, C), return normalized output

    def backward(self, grad_output):
        # input grad_output: (B, T, C), gradient of loss with respect to output
        assert grad_output.dim() == 3, f"Expected 3D grad_output, got {grad_output.dim()}D grad_output"
        B, T, C = grad_output.shape  # scalars, get dimensions of gradient tensor

        # compute intermediate gradient
        grad_x_hat = grad_output * self.gain.view(1, 1, C)  # (B, T, C), multiply gradient by gain parameter

        # compute approximate backpropagation
        x_hat = grad_output / (self.running_quad_mean.view(1, 1, C) + self.eps).sqrt()  # (B, T, C), normalize gradient by running quadratic mean
        quad_mean_grad = torch.sum(grad_x_hat * x_hat, dim=1, keepdim=True)  # (B, 1, C), compute gradient of quadratic mean
        grad_input = grad_x_hat - quad_mean_grad * x_hat / (2 * T * (self.running_quad_mean.view(1, 1, C) + self.eps).sqrt())  # (B, T, C), compute gradient with respect to input using approximate backpropagation

        return grad_input  # (B, T, C), return gradient with respect to input



# # Example usage
# num_features = 10
# x = torch.randn(20, 10, 50)  # Example input
# # model = ManualBatchNorm1d(50)
# # output = model(x)
# # print(output)
# model = BatchNorm(50)
# output = model(x)
# print(output)
# model = PowerNorm2(50)
# output = model(x)
# print(output)
# model = PowerNorm5(50)
# output = model(x)
# print(output)
