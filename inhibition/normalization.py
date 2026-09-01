import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition import init


class Square(nn.Module):
    """Point-wise square, the notebook's ``f(.)`` in the divisive-inhibition path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


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


class LayerNormalizeFunctionFA(torch.autograd.Function):
    """Identity forward with configurable LayerNorm-like backward ablations."""

    @staticmethod
    def forward(ctx, x, var, weights, no_backward, ln_feedback, module):
        epsilon = 1e-5
        ctx.no_backward = no_backward
        ctx.ln_feedback = ln_feedback
        ctx.module = module
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        actual_var = x.var(dim=-1, keepdim=True, unbiased=False)
        ctx.actual_var = torch.sqrt(actual_var + epsilon)
        ctx.save_for_backward(x_centered, var, weights.to(x.device))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_backward:
            return grad_output, None, None, None, None, None

        x_centered, var, weights = ctx.saved_tensors
        _ = var  # kept for API compatibility with call sites
        D = x_centered.shape[-1]
        grad_input = grad_output

        if ctx.ln_feedback == "full":
            grad_mean = grad_output.mean(dim=-1, keepdim=True)
            x_hat = x_centered / ctx.actual_var
            dot = (grad_output * x_hat).mean(dim=-1, keepdim=True)
            grad_input = (grad_output - grad_mean - x_hat * dot) / ctx.actual_var

        elif ctx.ln_feedback == "center":
            grad_mean = grad_output.mean(dim=-1, keepdim=True)
            grad_input = grad_output - grad_mean

        elif ctx.ln_feedback == "fa_center":
            if ctx.module is not None:
                ctx.module.grad_norm_delta = grad_output.detach().clone()
            grad_mean = (weights * grad_output).sum(dim=-1, keepdim=True)
            grad_input = grad_output - grad_mean

        elif ctx.ln_feedback == "scale":
            grad_input = grad_output / ctx.actual_var

        elif ctx.ln_feedback == "decorrelate":
            dot = (grad_output * x_centered).sum(dim=-1, keepdim=True)
            grad_input = grad_output - x_centered * dot / D

        return grad_input, None, None, None, None, None


class LayerNormFeedbackAblation(nn.Module):
    """Configurable backward-only LayerNorm-style gradient transform.

    Forward returns identity on ``x``. Backward can emulate or ablate pieces of
    the LayerNorm Jacobian through ``ln_feedback``:
    ``full``, ``center``, ``fa_center``, ``scale``, ``decorrelate``.
    """

    def __init__(self, ln_feedback="full", no_backward=False):
        super().__init__()
        self.ln_feedback = ln_feedback
        self.no_backward = no_backward
        self.grad_norm_delta = None

    def forward(self, x, var=None, weights=None):
        if var is None:
            var = x.var(dim=-1, keepdim=True, unbiased=False)
        if weights is None:
            weights = torch.ones_like(x) / x.shape[-1]
        return LayerNormalizeFunctionFA.apply(
            x,
            var,
            weights,
            self.no_backward,
            self.ln_feedback,
            self,
        )


class ParametrizedLayerNorm(nn.Module):
    """Predict scalar mean/variance from ``(x_t, h_prev)`` and normalize ``pre_act``.

    The predictors mirror the notebook's divisive-recurrent scheme: the mean is a
    single linear map (subtractive inhibition) and the variance is
    ``B_EI @ (B_IX @ feat)²`` (a ``Linear → square → average`` divisive stack).
    Calling :meth:`init_from_rnn_weights` with the parent RNN's weights sets these
    to reproduce ``LayerNorm`` exactly at initialization; the aux loss then keeps
    them LayerNorm-like as the RNN weights drift during training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        eps: float = 1e-5,
        stats_hidden_size: int | None = None,
        freeze_ei: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.freeze_ei = freeze_ei
        feat_dim = input_size + hidden_size
        # Number of divisive units. Default n_h matches the notebook's recurrent
        # example; ``W_eff`` is rank <= n_h - 1 so the last singular value is ~0.
        stats_hidden_size = stats_hidden_size or hidden_size
        self.mean_net = nn.Linear(feat_dim, 1)
        self.var_net = nn.Sequential(
            nn.Linear(feat_dim, stats_hidden_size),
            Square(),
            nn.Linear(stats_hidden_size, 1),
        )

        # Mirror the dense INormLayer's ``freeze_ei``: the divisive readout
        # (``var_net[2]`` <-> ``U_EI``) is a fixed, non-negative uniform average of
        # the squared projections. Freezing it removes the runaway output-scale
        # axis that lets ``pred_var`` blow up to infinity, so only the projection
        # (``var_net[0]`` <-> ``U_IE``) is trained. ``init_from_rnn_weights`` sets
        # this readout to the uniform average ``1/n_h``.
        if freeze_ei:
            for p in self.var_net[2].parameters():
                p.requires_grad_(False)

    def init_from_rnn_weights(
        self,
        W_XE: torch.Tensor,
        W_EE: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Init the stats predictors from the parent RNN's weights.

        Makes ``(pre_act - pred_mean) / sqrt(pred_var)`` equal ``LayerNorm(pre_act)``
        at initialization (see the notebook's divisive-recurrent derivation).
        """
        n_div = self.var_net[0].out_features
        init.parametrized_ln_mean_weight(self.mean_net, W_XE, W_EE, bias)
        init.parametrized_ln_var_weight(
            self.var_net[0], self.var_net[2], W_XE, W_EE, bias, n_div=n_div
        )

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
        # var_net is Linear -> square -> (non-negative) average, so its output is
        # already >= 0; no softplus needed (softplus would break the exact match).
        pred_var = self.var_net(feat) + self.eps
        return pred_mean, pred_var

    def aux_loss(
        self,
        pred_mean: torch.Tensor,
        pred_var: torch.Tensor,
        pre_act: torch.Tensor,
    ) -> torch.Tensor:
        """Match the predicted-stats normalization directly to true LayerNorm.

        MSE between ``(pre_act - pred_mean) / sqrt(pred_var)`` and
        ``LayerNorm(pre_act)``. Exactly 0 at init (the predicted stats reproduce the
        LayerNorm mean/variance) and well-posed on zero-variance (blank) rows, where
        both the prediction and the target are 0.

        Add this to your main loss at training time. ``pre_act`` is detached so the
        aux objective only trains the stats predictor; ``x_t`` / ``h_prev`` are
        detached before ``_predict_stats`` for the same reason.
        """
        x = pre_act.detach()
        pred_norm = (x - pred_mean) / torch.sqrt(pred_var)
        target = F.layer_norm(x, x.shape[-1:], eps=self.eps)
        return F.mse_loss(pred_norm, target)

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


class MeanNormalizeFunction(torch.autograd.Function):
    """Subtract the feature mean, with an optional backward (or forward) bypass."""

    @staticmethod
    def forward(ctx, x, no_backward, no_forward=False):
        ctx.no_backward = no_backward
        if no_forward:
            return x
        return x - x.mean(dim=-1, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_backward:
            return grad_output, None, None
        return grad_output - grad_output.mean(dim=-1, keepdim=True), None, None


class MeanNormalize(nn.Module):
    """``x - mean(x)`` over the feature dim.

    ``no_backward`` routes the gradient around the operation (the "detached
    norm" / LN-minus condition); ``no_forward`` makes the forward an identity so
    only the backward transform is applied.
    """

    def __init__(self, no_backward: bool = False, no_forward: bool = False):
        super().__init__()
        self.no_backward = no_backward
        self.no_forward = no_forward

    def forward(self, x):
        return MeanNormalizeFunction.apply(x, self.no_backward, self.no_forward)


class DivisiveNormalizeFunction(torch.autograd.Function):
    """Divide by the feature std (no centering), with backward/forward bypasses."""

    @staticmethod
    def forward(ctx, x, no_backward, no_forward=False):
        eps = 1e-5
        sigma = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + eps)
        ctx.save_for_backward(x, sigma)
        ctx.no_backward = no_backward
        if no_forward:
            return x
        return x / sigma

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_backward:
            return grad_output, None, None
        x, sigma = ctx.saved_tensors
        n = x.shape[-1]
        mu = x.mean(dim=-1, keepdim=True)
        dot = (grad_output * x).sum(dim=-1, keepdim=True)
        grad_input = grad_output / sigma - (x - mu) * dot / (sigma**3 * n)
        return grad_input, None, None


class DivisiveNormalize(nn.Module):
    """``x / sqrt(var(x) + eps)`` over the feature dim. See :class:`MeanNormalize`."""

    def __init__(self, no_backward: bool = False, no_forward: bool = False):
        super().__init__()
        self.no_backward = no_backward
        self.no_forward = no_forward

    def forward(self, x):
        return DivisiveNormalizeFunction.apply(x, self.no_backward, self.no_forward)


class LayerNormalizeFunction(torch.autograd.Function):
    """Full LayerNorm (no affine), with backward/forward bypasses."""

    @staticmethod
    def forward(ctx, x, no_backward, no_forward=False):
        eps = 1e-5
        mu = x.mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + eps)
        ctx.save_for_backward(x, mu, sigma)
        ctx.no_backward = no_backward
        if no_forward:
            return x
        return (x - mu) / sigma

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_backward:
            return grad_output, None, None
        x, mu, sigma = ctx.saved_tensors
        y = (x - mu) / sigma
        d = x.shape[-1]
        grad_mean = grad_output.mean(dim=-1, keepdim=True)
        dot = (grad_output * y).sum(dim=-1, keepdim=True)
        grad_input = (grad_output - grad_mean - y * dot / d) / sigma
        return grad_input, None, None


class LayerNormalizeCustom(nn.Module):
    """LayerNorm whose backward can be detached independently of its forward.

    ``no_backward=False`` is the paper's LN+ (normalization shapes both activity
    and error signals); ``no_backward=True`` is LN- (forward normalization only).
    """

    def __init__(self, no_backward: bool = False, no_forward: bool = False):
        super().__init__()
        self.no_backward = no_backward
        self.no_forward = no_forward

    def forward(self, x):
        return LayerNormalizeFunction.apply(x, self.no_backward, self.no_forward)
