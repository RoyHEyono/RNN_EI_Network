"""
Training helpers for NeuroGym supervised RNNs.

Follows the flow from the NeuroGym PyTorch supervised example (Dataset batching,
CrossEntropy over flattened time × batch, optional trial-wise evaluation):
https://neurogym.github.io/neurogym/latest/examples/supervised_learning_pytorch/
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


def require_neurogym() -> None:
    try:
        import neurogym  # noqa: F401
    except ImportError as e:
        print(
            "neurogym is required. Install with:\n"
            "  pip install 'rnn-ei-network[neurogym]'\n"
            "or: pip install neurogym",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def configure_neurogym_warnings() -> None:
    # NeuroGym's Dataset still uses env.seed / env.new_trial / env.ob / env.gt through
    # Gymnasium's deprecated wrapper forwarding; tasks omit render_modes in metadata.
    for pat in (
        r".*env\.seed to get variables from other wrappers.*",
        r".*env\.new_trial to get variables from other wrappers.*",
        r".*env\.ob to get variables from other wrappers.*",
        r".*env\.gt to get variables from other wrappers.*",
        r".*environment creator metadata doesn't include `render_modes`.*",
    ):
        warnings.filterwarnings("ignore", category=UserWarning, message=pat)


def ng_inputs_labels_to_torch(
    inputs_np: np.ndarray,
    labels_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeuroGym returns (seq, batch, ob); use batch-first tensors (batch, seq, …)."""
    inputs = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
    inputs = inputs.transpose(0, 1).contiguous()
    labels = labels.transpose(0, 1).contiguous()
    return inputs, labels


def trial_eval_accuracy(model: nn.Module, env: Any, device: torch.device, num_trial: int) -> float:
    """Match the doc example: single-trial rollouts, compare last-step choice to ``gt[-1]``."""
    model.eval()
    perf = 0
    with torch.no_grad():
        for _ in range(num_trial):
            env.get_wrapper_attr("new_trial")()
            ob = env.get_wrapper_attr("ob")
            gt = env.get_wrapper_attr("gt")
            ob = ob[:, np.newaxis, :]
            inputs = torch.from_numpy(ob).to(device=device, dtype=torch.float32).transpose(0, 1)
            logits = model(inputs)
            pred = logits.argmax(dim=-1).cpu().numpy()
            perf += int(gt[-1] == pred[0, -1])
    return perf / num_trial


def pretrain_parametrized_layer_norm(
    model: nn.Module,
    dataset: Any,
    optimizer_norm: optim.Optimizer,
    steps: int,
    device: torch.device,
    log_interval: int = 100,
    use_wandb: bool = False,
) -> float:
    """Calibrate SimpleEERNN's ParametrizedLayerNorm on fresh dataset batches.

    Only ``optimizer_norm.step()`` is called here, so W_XE/W_EE/bias/head stay at their
    initial values; ``ParametrizedLayerNorm.aux_loss`` already detaches ``pre_act``/``x_t``/
    ``h_prev``, so no RNN gradients would reach those params here regardless. Tracks the
    lowest-aux-loss ``state_dict`` of the norm module seen and restores it before returning,
    matching the calibration pattern in ``test/rnn_test.py``.
    """
    rnn = model.rnn
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for i in range(steps):
        inputs_np, labels_np = dataset()
        inputs, _ = ng_inputs_labels_to_torch(inputs_np, labels_np, device)

        optimizer_norm.zero_grad(set_to_none=True)
        _, _, aux_losses = rnn(inputs)
        total_aux = sum(aux_losses)
        total_aux.backward()
        optimizer_norm.step()

        loss_val = float(total_aux.item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in rnn.layer_norm.state_dict().items()}
        if (i + 1) % log_interval == 0:
            print(f"[param-ln pretrain] step {i + 1}  aux_loss: {loss_val:.6f}  best: {best_loss:.6f}")
            if use_wandb:
                # No explicit `step=` here: the main loop below logs with explicit step=i+1
                # starting from log_interval, and wandb requires non-decreasing step values.
                wandb.log({"pretrain/aux_loss": loss_val, "pretrain/best_aux_loss": best_loss})

    if best_state is not None:
        rnn.layer_norm.load_state_dict(best_state)
    return best_loss


def train_supervised_steps(
    args: Any,
    model: nn.Module,
    dataset: Any,
    env: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    optimizer_norm: optim.Optimizer | None = None,
    aux_loss_weight: float = 1.0,
) -> None:
    # Best (lowest) losses within the current log window; reset every log_interval
    # so they reflect recent training rather than a stale global minimum.
    best_loss = float("inf")
    best_aux_loss = float("inf")
    best_trial_acc = float("-inf")
    trial_accuracies: list[float] = []
    for i in range(args.epochs):
        model.train()
        inputs_np, labels_np = dataset()
        inputs, labels = ng_inputs_labels_to_torch(inputs_np, labels_np, device)

        optimizer.zero_grad(set_to_none=True)
        if optimizer_norm is not None:
            optimizer_norm.zero_grad(set_to_none=True)
        logits = model(inputs)
        b, t, c = logits.shape
        loss = criterion(logits.reshape(b * t, c), labels.reshape(b * t))
        aux_loss = getattr(model, "last_aux_loss", None)
        if aux_loss is not None:
            loss = loss + aux_loss_weight * aux_loss
        loss.backward()
        optimizer.step()
        if optimizer_norm is not None:
            optimizer_norm.step()

        trial_acc: float | None = None
        if args.eval_trials > 0:
            trial_acc = trial_eval_accuracy(model, env, device, args.eval_trials)
            trial_accuracies.append(trial_acc)
            best_trial_acc = max(best_trial_acc, trial_acc)

        best_loss = min(best_loss, float(loss.item()))
        if aux_loss is not None:
            best_aux_loss = min(best_aux_loss, float(aux_loss.item()))
        if getattr(args, "wandb", False):
            payload: dict[str, Any] = {"train/best_loss": best_loss}
            if aux_loss is not None:
                payload["train/best_aux_loss"] = best_aux_loss
            if trial_acc is not None:
                payload["eval/trial_accuracy"] = best_trial_acc
                payload["eval/trial_accuracy_auc"] = float(np.trapezoid(trial_accuracies))
            wandb.log(payload, step=i + 1)

        if (i + 1) % args.log_interval == 0:
            line = f"step {i + 1}  best_loss_last_{args.log_interval}: {best_loss:.5f}"
            if aux_loss is not None:
                line += f"  best_aux_loss_last_{args.log_interval}: {best_aux_loss:.5f}"
            if trial_acc is not None:
                line += (
                    f"  best_trial_acc_last_{args.log_interval} ({args.eval_trials} trials): {best_trial_acc:.4f}"
                )
            print(line)
            best_loss = float("inf")
            best_aux_loss = float("inf")
            best_trial_acc = float("-inf")
