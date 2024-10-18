import torch
import torch.nn as nn


# Custom Activation Variations
class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)

class PiecewiseLearnableActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_of_points = 7

        # Initialize learnable parameters for x and y values of intermediate points
        self.x_vals = nn.Parameter(torch.linspace(-2, 2, self.num_of_points + 2)[1:-1])  # Exclude -2 and 2

        # Initialize y_vals using GELU output for corresponding x_vals
        gelu = nn.GELU()
        self.y_vals = nn.Parameter(gelu(self.x_vals))

    def forward(self, x):
        # Create a piecewise linear function
        result = torch.zeros_like(x)

        # Leftmost segment (-2 <= x < x_vals[0])
        result = torch.where(x < self.x_vals[0], 0, result)  # x = -2 -> y = 0

        # Intermediate segments (x_vals[i] <= x < x_vals[i+1])
        for i in range(self.num_of_points - 1):
            slope = (self.y_vals[i + 1] - self.y_vals[i]) / (self.x_vals[i + 1] - self.x_vals[i])
            intercept = self.y_vals[i] - slope * self.x_vals[i]
            segment = slope * x + intercept
            result = torch.where((x >= self.x_vals[i]) & (x < self.x_vals[i + 1]), segment, result)

        # Segment before the last (x_vals[-1] <= x < 2)
        slope = (2 - self.y_vals[-1]) / (2 - self.x_vals[-1])
        intercept = self.y_vals[-1] - slope * self.x_vals[-1]
        segment = slope * x + intercept
        result = torch.where((x >= self.x_vals[-1]) & (x < 2), segment, result)

        # Rightmost segment (x >= 2)
        result = torch.where(x >= 2, x, result)  # x = y for x >= 2

        return result

activation_dictionary = {
    "celu": nn.CELU(),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "glu": nn.GLU(),
    "leaky_relu": nn.LeakyReLU(),
    "mish": nn.Mish(),
    "piecewise": PiecewiseLearnableActivation(),
    "prelu": nn.PReLU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "rrelu": nn.RReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "squared_relu": SquaredReLU(),
    "tanh": nn.Tanh(),
}
