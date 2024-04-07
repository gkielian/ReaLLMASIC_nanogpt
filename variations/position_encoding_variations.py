import torch
import torch.nn as nn

def safe_log(x, eps=1e-6):
    """
    Computes a safe logarithm of x by clamping the minimum value to eps.
    This prevents log(0) and log(negative) scenarios.
    """
    return torch.log(torch.clamp(x, min=eps))

class FIRE(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512.0, eps=1e-6):
        super(FIRE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, num_heads)
        )
        self.c = nn.Parameter(torch.tensor(init_c, dtype=torch.float))
        self.init_L = nn.Parameter(torch.tensor(init_L, dtype=torch.float), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0, dtype=torch.float))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        # Apply absolute value and ensure positive before log
        abs_rel_distance = torch.abs(rel_distance) + self.eps

        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None] + self.eps  # Ensure pos_normalizer is never zero

        # Use safe log operation
        log_rel_distance = torch.log(abs_rel_distance * self.c + self.eps)
        log_pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + self.eps)

        normalized_distance = log_rel_distance - log_pos_normalizer  # Subtraction instead of division

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        return fire_bias

class FIRE_safe(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512.0, eps=1e-3):
        super(FIRE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, num_heads)
        )
        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None]

        # Using safe_log for logging operations
        rel_distance = safe_log(torch.abs(self.c * rel_distance) + 1, self.eps)
        pos_normalizer = safe_log(torch.abs(self.c * pos_normalizer) + 1, self.eps)

        normalized_distance = rel_distance / pos_normalizer

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        return fire_bias

class FIRE_(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512.0, eps=1e-3):
        super(FIRE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, num_heads)
        )
        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        seq_length = x.size(1)  # Adjusted to correct dimension
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None]

        # Check for nan
        if torch.isnan(rel_distance).any():
            print("Nan found in rel_distance")

        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Check for nan after transformations
        if torch.isnan(rel_distance).any():
            print("Nan found in log-transformed rel_distance")
        if torch.isnan(pos_normalizer).any():
            print("Nan found in log-transformed pos_normalizer")

        normalized_distance = rel_distance / pos_normalizer

        if torch.isnan(normalized_distance).any():
            print("Nan found in normalized_distance")

        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)

        if torch.isnan(fire_bias).any():
            print("Nan found in fire_bias")

        return fire_bias


class FIRE2(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512.0,
                 eps=1e-3):
        """
        FIRE attention bias module.

         Args:
         num_heads: number of attention heads.
         mlp_width: Width of MLP.
         init_c: initial value of log transformation parameter
         init_L: initial value of thresholding parameter
         eps: small constant for numerical stability
        """
        super(FIRE, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, num_heads)
        )

        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        # Learn a multiplier to L
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Compute FIRE attention bias.

        Args:
        x: input sequence,
        shape [bsz, num_heads, seq_len, hidden_dim]

        Returns:
        attention bias,
        shape [1, num_heads, seq_len, seq_len]
        """
        seq_length = x.size(1)
        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        # Thresholding the normalizer
        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None]

        # Amplifying differences among local positions
        # with log transform
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation
        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)
        return fire_bias


class FIRE2kOu(nn.Module):
    def __init__(self, num_heads=6, mlp_width=32, init_c=0.1, init_L=256.0, eps=1e-6):
        super(FIRE, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width), nn.ReLU(), nn.Linear(mlp_width, num_heads)
        )

        self.c = nn.Parameter(torch.tensor(init_c))
        self.init_L = nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Compute FIRE attention bias.

        Args:
            x: Input tensor with shape [batch_size, seq_len, embedding_dim]

        Returns:
            Attention bias of shape [batch_size, num_heads, seq_len, seq_len]
        """

        batch_size, seq_len, _ = x.shape

        # Generate relative positions
        positions = torch.arange(seq_len, dtype=torch.float, device=x.device)
        rel_distance = (
            positions[None, :, None] - positions[None, None, :]
        )  # [1, seq_len, seq_len]

        # Thresholding
        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(rel_distance, threshold.expand_as(rel_distance))

        # Amplification with log transform
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Normalization and MLP
        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(
            normalized_distance.unsqueeze(-1)
        )  # [1, seq_len, seq_len, num_heads]

        # Expand batch dimension and permute for proper shape
        fire_bias = fire_bias.expand(batch_size, -1, -1, -1).permute(0, 3, 1, 2)

        return fire_bias


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd

        # Register frequencies directly as buffers
        self.register_buffer(
            "freq_left",
            (10000 ** (torch.arange(0, self.dim // 2).float() / self.dim // 2)),
        )
        self.register_buffer(
            "freq_right",
            (10000 ** (torch.arange(0, self.dim // 2).float() / self.dim // 2)),
        )

    def forward(self, x):
        seq_len = x.shape[-2]
        device = x.device

        t = torch.arange(seq_len, device=device)

        # Get separate frequencies for left and right
        freqs_left = torch.einsum("i,j->ij", t, self.freq_left)
        freqs_right = torch.einsum("i,j->ij", t, self.freq_right)

        # Apply frequencies
        x_left, x_right = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Combine the left and right parts back
        x = torch.cat([x_left, x_right], dim=-1)

        return x


class ShortRope(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n = config.shortrope_length
        self.dim = config.n_embd

        # Generate freqs of size n rather than full dim
        self.register_buffer(
            "freq_left", (10000 ** (torch.arange(0, self.n // 2).float() / self.n // 2))
        )
        self.register_buffer(
            "freq_right",
            (10000 ** (torch.arange(0, self.n // 2).float() / self.n // 2)),
        )

    def forward(self, x):
        # Step 1: Get the input tensor shape
        batch_size, seq_len, _ = x.shape

        # Step 2: Split the input tensor into unrotated and rotated sections
        x_unrotated = x[..., : -self.n]  # All but the last n dimensions
        x_rotated = x[..., -self.n :]  # Only the last n dimensions

        # Step 3: Generate rotation frequencies
        t = torch.arange(self.n, device=x.device)
        freqs_left = torch.einsum("i,j->ij", t, self.freq_left)
        freqs_right = torch.einsum("i,j->ij", t, self.freq_right)

        # Calculate how many times to repeat freqs along the sequence length
        num_repeats = seq_len // self.n + int(seq_len % self.n != 0)

        # Repeat the frequency tensors to match the sequence length
        freqs_left = freqs_left.repeat(batch_size, num_repeats, 1)
        freqs_right = freqs_right.repeat(batch_size, num_repeats, 1)

        # Trim the excess elements so the freqs tensors match the sequence length
        freqs_left = freqs_left[:, :seq_len, :]
        freqs_right = freqs_right[:, :seq_len, :]

        # Step 4: Process the x_rotated section
        x_left = x_rotated[..., : self.n // 2]
        x_right = x_rotated[..., self.n // 2 :]

        # Apply the cosine and sine rotations
        x_left = x_left * freqs_left.cos() - x_right * freqs_left.sin()
        x_right = x_left * freqs_right.sin() + x_right * freqs_right.cos()

        # Invert the order of the right tensor's last dimension and negate it
        x_right = torch.flip(x_right, dims=[-1]) * -1

        # Combine the left and right rotated sections
        x_rotated = torch.cat([x_left, x_right], dim=-1)

        # Step 5: Combine the rotated and unrotated sections
        x = torch.cat([x_unrotated, x_rotated], dim=-1)

        return x
