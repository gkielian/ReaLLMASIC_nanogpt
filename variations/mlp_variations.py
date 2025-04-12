# variations/mlp_variations.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from variations.activation_variations import activation_dictionary
from variations.linear_variations import linear_dictionary
from quantization.quantize import fake_quantize_act
from quantization.quant_utils import set_variant, create_activation_buffers
from torch.linalg import matrix_exp


class OriginalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        # Select "mlp variant"
        self.mlp_variant = config.mlp_variant
        self.use_mlp_res = config.mlp_res

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        # Instantiate Linear Layers
        if self.mlp_variant == "mlp":
            self.c_fc = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"], bias=config.bias)
            self.c_proj = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"], bias=config.bias)
        elif self.mlp_variant == "swiglu":
            self.c_fc_in1 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
            self.c_fc_in2 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
            self.c_fc_out = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None, mlp_res=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        if self.mlp_variant == "mlp":
            x = self.c_fc(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

            x = self.activation_variant(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

            # MLP Residual
            if self.use_mlp_res:
                if mlp_res is None:
                    mlp_res = torch.zeros_like(x)
                mlp_res = x + mlp_res
                x = mlp_res


            x = self.c_proj(x)

        elif self.mlp_variant == "swiglu":
            x_in1 = self.c_fc_in1(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

            x_in1 = self.activation_variant(x_in1)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

            x_in2 = self.c_fc_in2(x)
            x_out = x_in1 * x_in2

            # MLP Residual on the x_out
            if self.use_mlp_res:
                if mlp_res is None:
                    mlp_res = torch.zeros_like(x_out)
                x_out = mlp_res + x_out
                mlp_res = x_out

            x = self.c_fc_out(x_out)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x, mlp_res

class Swiglu(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        self.c_fc_in1 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
        self.c_fc_in2 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
        self.c_fc_out = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"])

        self.mlp_res_gate = self.linear_variant_mlp_up(config.mlp_expansion_factor * config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None, mlp_res=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        x_in1 = self.c_fc_in1(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

        x_in1 = self.activation_variant(x_in1)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

        x_in2 = self.c_fc_in2(x)
        x_out = x_in1 * x_in2

        # MLP Residual on the x_out
        if mlp_res is None:
            mlp_res = torch.zeros_like(x_out)
        x_out = mlp_res + x_out
        mlp_res = x_out

        x = self.c_fc_out(x_out)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x, mlp_res

class KanMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.kan = linear_dictionary["kan"](config.n_embd, config.n_embd, config=config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        x = self.kan(x)
        x = self.dropout(x)

        return x

class LearnedRotationMLP(nn.Module):
    """
    MLP variation using a single learned rotation.
    Inspired by "On Learning Rotations" (Arora, 2009), using the exponential map
    from a skew-symmetric matrix. Simpler structure like KanMLP.
    Ignores quantization internally for simplicity.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        # Learnable matrix A. Skew-symmetry is enforced in the forward pass.
        # Initialize near zero for near-identity initial rotation.
        self.A = nn.Parameter(torch.randn(config.n_embd, config.n_embd) * 0.01)

        if config.learned_rotation_scaling:
            self.scaling_factor = nn.Parameter(torch.tensor([1.0]))
        else:
            self.scaling_factor = 1.0

        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None, mlp_res=None): # Match signature, ignore mlp_res
        # Enforce skew-symmetry
        S = self.A - self.A.transpose(-1, -2)
        # Compute rotation matrix using matrix exponential
        R = matrix_exp(S) # R is now in SO(n_embd)
        # Apply rotation
        x = x @ R # x is (B, T, C), R is (C, C)
        x = x * self.scaling_factor
        # Apply dropout
        x = self.dropout_layer(x)
        # Return x and None for mlp_res to match expected return signature of other MLPs
        return x, None

mlp_dictionary = {
    "mlp": OriginalMLP,
    "swiglu": Swiglu,
    "kan": KanMLP,
    "learned_rotation": LearnedRotationMLP,
    }

def get_mlp_instance(config):
    mlp_type = config.mlp_variant
    mlp_class = mlp_dictionary.get(mlp_type)
    if mlp_class is None:
        raise ValueError(f"Unsupported MLP variant: {mlp_type}")
    return mlp_class(config)

