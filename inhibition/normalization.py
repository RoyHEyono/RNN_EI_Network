import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.init import calc_ln_mu_sigma


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
    """Predict scalar mean/variance from ``(x_t, h_prev)`` and normalize ``pre_act``.

    Mean and variance are each predicted by their own small MLP rather than a
    shared linear projection, giving the stats predictor enough capacity to
    fit nonlinear mean/variance surfaces (a single linear map converges slowly
    on anything but an already-linear target).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        eps: float = 1e-5,
        stats_hidden_size: int | None = None,
    ):
        super().__init__()
        self.eps = eps
        feat_dim = input_size + hidden_size
        stats_hidden_size = stats_hidden_size or feat_dim
        self.mean_net = nn.Sequential(
            nn.Linear(feat_dim, stats_hidden_size),
            nn.ReLU(),
            nn.Linear(stats_hidden_size, 1),
        )
        self.var_net = nn.Sequential(
            nn.Linear(feat_dim, stats_hidden_size),
            nn.ReLU(),
            nn.Linear(stats_hidden_size, 1),
        )
        for net in (self.mean_net, self.var_net):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    self._lognormal_weight_(layer.weight)

    @staticmethod
    def _lognormal_weight_(
        weight: torch.Tensor, numerator: float = 2.0, k: float = 1.0
    ) -> None:
        """Lognormal init matching ``inhibition.init``'s excitatory scheme.

        Fan-in scaling only: ``excitatory_weight``'s ``(ne - 1)`` factor divides by
        zero on the single-unit output layers here. Marks ``clamp=True`` so callers
        keep these weights non-negative after optimizer steps.
        """
        _, n_in = weight.shape
        target_std = (numerator / n_in) ** 0.5
        mu, sigma = calc_ln_mu_sigma(target_std * k, target_std**2)
        with torch.no_grad():
            weight.log_normal_(mean=float(mu), std=float(sigma))
        weight.clamp = False

    def _clamp_weights(self) -> None:
        with torch.no_grad():
            for p in self.parameters():
                if getattr(p, "clamp", False):
                    p.clamp_(min=0)

    def _predict_stats(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.cat([x_t, h_prev], dim=-1)
        pred_mean = self.mean_net(feat)
        pred_var = F.softplus(self.var_net(feat)) + self.eps
        return pred_mean, pred_var

    def aux_loss(
        self,
        pred_mean: torch.Tensor,
        pred_var: torch.Tensor,
        pre_act: torch.Tensor,
    ) -> torch.Tensor:
        """Push predicted-normalized activations toward mean 0 and variance 1.

        Add this to your main loss at training time. At inference you use only
        the predicted stats, so this is what makes the module *approximate*
        LayerNorm rather than apply an arbitrary affine per step. ``pre_act``
        is detached so the aux objective only trains the stats predictor;
        ``x_t`` / ``h_prev`` are detached before ``_predict_stats`` for the
        same reason.
        """
        x = pre_act.detach()
        pred_norm = (x - pred_mean) / torch.sqrt(pred_var)
        # Encourage LayerNorm-like stats: feature-wise mean ~ 0, var ~ 1.
        mean = pred_norm.mean(dim=-1)
        var = pred_norm.var(dim=-1, unbiased=False)
        return mean.pow(2).mean() + (var - 1).pow(2).mean()

    def measure_layer_norm_mse(
        self,
        pred_mean: torch.Tensor,
        pred_var: torch.Tensor,
        pre_act: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted-stats norm and true ``LayerNorm(pre_act)``.

        Diagnostic only — not used as a training objective.
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
