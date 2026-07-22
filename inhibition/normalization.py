import torch
import torch.nn as nn
import torch.nn.functional as F


class layer_norm_linear_ste(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, x):
        # 1. The 'Linear' path
        linear_out = x

        # 2. The 'LayerNorm' path
        ln_out = self.layer_norm(x)

        # 3. The Hijack:
        # Forward is linear_out, Backward is ln_out's gradient
        return ln_out + (linear_out - ln_out).detach()


class ParametrizedLayerNorm(nn.Module):
    """Predict scalar mean/variance from ``(x_t, h_prev)`` and normalize ``pre_act``."""

    def __init__(self, input_size: int, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.proj = nn.Linear(input_size + hidden_size, 2)

    def forward(
        self, pre_act: torch.Tensor, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        stats = self.proj(torch.cat([x_t, h_prev], dim=-1))  # (B, 2)
        mean = stats[..., :1]
        var = F.softplus(stats[..., 1:]) + self.eps
        return (pre_act - mean) / torch.sqrt(var)
