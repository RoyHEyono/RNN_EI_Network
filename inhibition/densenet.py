"""Dense Fashion-MNIST networks for the inhibitory-normalization paper.

Three variants, all with two hidden layers (``fc0``, ``fc1``) and an
:class:`~inhibition.dense.EiDenseLayer` readout:

* :class:`EDenseNet`  -- excitatory only (Fig. 2b's "E-only").
* :class:`EIDenseNet` -- standard E/I, optionally with hard-coded normalization.
* :class:`INormDenseNet` -- inhibition trained to normalize (Figs. 3-7).

Each net logs per-layer activity moments (and, when alignment tracking is on,
cosine similarities against LayerNorm) under the same wandb key names the
paper's figure notebooks query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from inhibition.dense import EDenseLayer, EiDenseLayer, INormLayer
from inhibition.normalization import (
    DivisiveNormalize,
    LayerNormalizeCustom,
    MeanNormalize,
)

MNIST_FLAT = 28 * 28
N_CLASSES = 10

NO_NORMALIZE = 0
MEAN_NORMALIZE = 1
VAR_NORMALIZE = 2
LN_NORMALIZE = 3


def build_norm(norm_type: int, no_backward: bool):
    """Hard-coded normalization applied between layer and nonlinearity.

    ``no_backward=True`` is the paper's LN- condition: normalization shapes the
    forward activity but the error signal is routed around it.
    """
    if norm_type == NO_NORMALIZE:
        return None
    if norm_type == MEAN_NORMALIZE:
        return MeanNormalize(no_backward=no_backward)
    if norm_type == VAR_NORMALIZE:
        return DivisiveNormalize(no_backward=no_backward)
    if norm_type == LN_NORMALIZE:
        return LayerNormalizeCustom(no_backward=no_backward)
    raise ValueError(f"unknown norm_type {norm_type}")


def norm_type_from_flags(normtype: int, divisive_norm: int, layer_norm: int) -> int:
    """Mirror the reference repo's flag precedence: mean > divisive > LN > none."""
    if normtype:
        return MEAN_NORMALIZE
    if divisive_norm:
        return VAR_NORMALIZE
    if layer_norm:
        return LN_NORMALIZE
    return NO_NORMALIZE


class _DenseNetBase(nn.Module):
    """Shared plumbing: hidden-layer names, forward loop and wandb logging."""

    def __init__(self, num_layers: int = 1, wandb_log: bool = False):
        super().__init__()
        # ``num_layers=1`` gives fc0 + fc1, i.e. the paper's two hidden layers.
        self.num_layers = num_layers
        self.hidden_names = [f"fc{i}" for i in range(num_layers + 1)]
        self.wandb_log = wandb_log
        self.register_eval = False
        self.relu = nn.ReLU()
        self._layer_inputs: list[torch.Tensor] = []

    def hidden_layers(self):
        return [getattr(self, name) for name in self.hidden_names]

    def set_eval_logging(self, flag: bool) -> None:
        """Switch the metric prefix between ``train_`` and ``eval_``."""
        self.register_eval = flag

    def _log_layer(self, name: str, layer: nn.Module, out: torch.Tensor) -> None:
        if not self.wandb_log:
            return
        import wandb

        prefix = "eval" if self.register_eval else "train"
        payload = {
            f"{prefix}_{name}_mu": out.mean(dim=-1).mean().item(),
            f"{prefix}_{name}_var": out.var(dim=-1, unbiased=False).mean().item(),
        }
        if not self.register_eval and getattr(layer, "track_alignment", False):
            payload[f"gradient_alignment_{name}"] = layer.gradient_alignment_val
            payload[f"output_alignment_{name}"] = layer.output_alignment_val
        if self.register_eval and getattr(layer, "ln_feedback", None) == "fa_center":
            lam = _max_gradient_eigenvalue(layer)
            if lam is not None:
                payload[f"eval_{name}_gradient_eigenvalue_max"] = lam
        wandb.log(payload, commit=False)

    def _apply_norm(self, x):
        return x

    def forward(self, x, return_layer_inputs: bool = False):
        h = torch.flatten(x, 1)
        self._layer_inputs = []
        for name in self.hidden_names:
            layer = getattr(self, name)
            self._layer_inputs.append(h)
            z = layer(h)
            self._log_layer(name, layer, z)
            h = self.relu(self._apply_norm(z))
        logits = self.fc_output(h)
        if return_layer_inputs:
            return logits, tuple(self._layer_inputs)
        return logits


def _max_gradient_eigenvalue(layer) -> float | None:
    """Largest eigenvalue of the covariance of the layer's incoming error signal.

    Reads the delta stashed by ``LayerNormFeedbackAblation`` during the most
    recent backward pass (appendix panel for the lateral-inhibition condition).
    """
    grad_output = getattr(layer.grad_norm, "grad_norm_delta", None)
    if grad_output is None or grad_output.shape[0] < 2:
        return None
    centered = grad_output - grad_output.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (centered.shape[0] - 1)
    return torch.linalg.eigvalsh(cov.float()).max().item()


class EIDenseNet(_DenseNetBase):
    """Standard E/I network, optionally with hard-coded normalization."""

    def __init__(
        self,
        input_size: int = MNIST_FLAT,
        hidden_size: int = 234,
        output_size: int = N_CLASSES,
        num_layers: int = 1,
        norm_type: int = NO_NORMALIZE,
        normtype_detach: bool = False,
        track_alignment: bool = False,
        wandb_log: bool = False,
    ):
        super().__init__(num_layers=num_layers, wandb_log=wandb_log)
        in_dim = input_size
        for name in self.hidden_names:
            setattr(
                self,
                name,
                EiDenseLayer(
                    in_dim,
                    hidden_size,
                    inh_ratio=max(1, int(0.1 * hidden_size)) / hidden_size,
                    track_alignment=track_alignment,
                ),
            )
            in_dim = hidden_size
        self.fc_output = EiDenseLayer(
            hidden_size, output_size, inh_ratio=max(1, int(0.1 * output_size)) / output_size
        )
        self.ln = build_norm(norm_type, no_backward=bool(normtype_detach))

    def _apply_norm(self, x):
        return x if self.ln is None else self.ln(x)


class EDenseNet(EIDenseNet):
    """Excitatory-only network: same layout with the inhibitory pathway ablated."""

    def __init__(
        self,
        input_size: int = MNIST_FLAT,
        hidden_size: int = 234,
        output_size: int = N_CLASSES,
        num_layers: int = 1,
        norm_type: int = NO_NORMALIZE,
        normtype_detach: bool = False,
        track_alignment: bool = False,
        wandb_log: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            norm_type=norm_type,
            normtype_detach=normtype_detach,
            track_alignment=track_alignment,
            wandb_log=wandb_log,
        )
        # Replace the hidden E/I layers with purely excitatory ones. The readout
        # stays an EiDenseLayer, as in the reference implementation.
        in_dim = input_size
        for name in self.hidden_names:
            setattr(self, name, EDenseLayer(in_dim, hidden_size, track_alignment=track_alignment))
            in_dim = hidden_size


class INormDenseNet(_DenseNetBase):
    """I-Norm network: inhibition is trained to normalize excitatory activity.

    Normalization happens inside each :class:`~inhibition.dense.INormLayer`, so
    there is no hard-coded norm between layers. Call :meth:`local_loss` after a
    forward pass to get the I-Norm objective for the inhibitory weights.
    """

    def __init__(
        self,
        input_size: int = MNIST_FLAT,
        hidden_size: int = 234,
        output_size: int = N_CLASSES,
        num_layers: int = 1,
        ln_feedback: str = "full",
        gradient_norm: bool = True,
        shunting: bool = True,
        track_alignment: bool = False,
        freeze_ei: bool = True,
        wandb_log: bool = False,
    ):
        super().__init__(num_layers=num_layers, wandb_log=wandb_log)
        in_dim = input_size
        for name in self.hidden_names:
            setattr(
                self,
                name,
                INormLayer(
                    in_dim,
                    hidden_size,
                    inh_ratio=max(1, int(0.1 * hidden_size)) / hidden_size,
                    ln_feedback=ln_feedback,
                    gradient_norm=gradient_norm,
                    shunting=shunting,
                    track_alignment=track_alignment,
                    freeze_ei=freeze_ei,
                ),
            )
            in_dim = hidden_size
        self.fc_output = EiDenseLayer(
            hidden_size, output_size, inh_ratio=max(1, int(0.1 * output_size)) / output_size
        )

    def local_loss(self):
        """I-Norm loss summed over hidden layers, plus per-layer LN-MSE diagnostics.

        Must be called after :meth:`forward`; it reuses the cached layer inputs.
        """
        if not self._layer_inputs:
            raise RuntimeError("call forward() before local_loss()")
        total = None
        diagnostics = {}
        for name, h_prev in zip(self.hidden_names, self._layer_inputs):
            moments, ln_mse = getattr(self, name).local_loss(h_prev)
            total = moments if total is None else total + moments
            diagnostics[name] = ln_mse
        return total, diagnostics
