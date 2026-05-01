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


def train_supervised_steps(
    args: Any,
    model: nn.Module,
    dataset: Any,
    env: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> None:
    running_loss = 0.0
    for i in range(args.epochs):
        model.train()
        inputs_np, labels_np = dataset()
        inputs, labels = ng_inputs_labels_to_torch(inputs_np, labels_np, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        b, t, c = logits.shape
        loss = criterion(logits.reshape(b * t, c), labels.reshape(b * t))
        loss.backward()
        optimizer.step()

        trial_acc: float | None = None
        if args.eval_trials > 0:
            trial_acc = trial_eval_accuracy(model, env, device, args.eval_trials)

        running_loss += float(loss.item())
        if (i + 1) % args.log_interval == 0:
            mean_loss = running_loss / args.log_interval
            line = f"step {i + 1}  mean_loss_last_{args.log_interval}: {mean_loss:.5f}"
            if trial_acc is not None:
                line += f"  trial_acc ({args.eval_trials} trials): {trial_acc:.4f}"
            print(line)
            if getattr(args, "wandb", False):
                payload: dict[str, Any] = {
                    "train/loss_batch_mean": loss.item(),
                    "train/loss_window_mean": mean_loss,
                }
                if trial_acc is not None:
                    payload["eval/trial_accuracy"] = trial_acc
                wandb.log(payload, step=i + 1)
            running_loss = 0.0
        elif getattr(args, "wandb", False) and trial_acc is not None:
            wandb.log({"eval/trial_accuracy": trial_acc}, step=i + 1)
