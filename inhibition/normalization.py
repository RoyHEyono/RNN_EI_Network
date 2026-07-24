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

    def _predict_stats(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.proj(torch.cat([x_t, h_prev], dim=-1))  # (B, 2)
        pred_mean = stats[..., :1]
        pred_var = F.softplus(stats[..., 1:]) + self.eps
        return pred_mean, pred_var

    def aux_loss(
        self,
        pred_mean: torch.Tensor,
        pred_var: torch.Tensor,
        pre_act: torch.Tensor,
    ) -> torch.Tensor:
        """Match predicted-normalized activations to true LayerNorm(pre_act).

        Add this to your main loss at training time. At inference you use only
        the predicted stats, so this is what makes the module *approximate*
        LayerNorm rather than apply an arbitrary affine per step. ``pre_act``
        is detached so the aux objective only trains the stats predictor;
        ``x_t`` / ``h_prev`` are detached before ``_predict_stats`` for the
        same reason.
        """
        x = pre_act.detach()
        pred_norm = (x - pred_mean) / torch.sqrt(pred_var)
        tgt = F.layer_norm(x, x.shape[-1:], eps=self.eps)
        return F.mse_loss(pred_norm, tgt)

    def forward(
        self, pre_act: torch.Tensor, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Detach context so aux (and STE-detached predicted path) cannot
        # send gradients into the RNN inputs that feed the stats predictor.
        pred_mean, pred_var = self._predict_stats(x_t.detach(), h_prev.detach())
        normalized = (pre_act - pred_mean) / torch.sqrt(pred_var)
        aux = self.aux_loss(pred_mean, pred_var, pre_act)
        # Forward: predicted-stats norm; backward: true LayerNorm Jacobian.
        ln_true = F.layer_norm(pre_act, pre_act.shape[-1:], eps=self.eps)
        out = ln_true + (normalized - ln_true).detach()
        return out, aux
