from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def training_loss_from_batch(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    *,
    local_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single-batch forward + local loss; returns scalar loss and detached metrics."""
    output, layer_inputs = model(data, return_layer_inputs=True)
    task_loss = F.nll_loss(output, target)
    local_moment = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
    local_ln_sum = 0.0
    for layer, h_prev in zip(model.inorm_layers(), layer_inputs):
        moments_term, ln_term = layer.local_loss(h_prev)
        local_moment = local_moment + moments_term
        local_ln_sum += ln_term
    loss = task_loss + local_loss_weight * local_moment
    metrics = {
        "task_loss": task_loss.detach().item(),
        "local_ln_sum": local_ln_sum,
    }
    return loss, metrics


def train_one_epoch(
    args: Any,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    *,
    verbose: bool = True,
) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss, metrics = training_loss_from_batch(
            model, data, target, local_loss_weight=args.local_loss_weight
        )
        loss.backward()
        optimizer.step()
        if verbose and batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (task {:.6f}, ln {:.6f})".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    metrics["task_loss"],
                    metrics["local_ln_sum"],
                )
            )
            if args.dry_run:
                break


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
) -> dict[str, float | int]:
    """Aggregate NLL and accuracy on the full loader (no prints)."""
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            n_correct += pred.eq(target.view_as(pred)).sum().item()
            n_total += target.shape[0]
    mean_loss = total_loss / n_total
    accuracy_pct = 100.0 * n_correct / n_total
    return {
        "loss": mean_loss,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy_pct": accuracy_pct,
    }


def format_eval_metrics(m: Mapping[str, float | int]) -> str:
    """Human-readable line matching the former evaluate() printout."""
    return (
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            m["loss"],
            m["n_correct"],
            m["n_total"],
            m["accuracy_pct"],
        )
    )
