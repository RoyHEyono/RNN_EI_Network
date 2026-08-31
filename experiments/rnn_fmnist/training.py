from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import wandb


criterion = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction="sum")


def train_one_epoch(
    args: Any,
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    *,
    optimizer_norm: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "task_loss": 0.0, "aux_loss": 0.0}
    n_examples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        if optimizer_norm is not None:
            optimizer_norm.zero_grad(set_to_none=True)

        logits = model(data)
        task_loss = criterion(logits, target)
        aux_loss = getattr(model, "last_aux_loss", None)
        loss = task_loss
        if aux_loss is not None:
            loss = loss + args.aux_loss_weight * aux_loss

        loss.backward()
        optimizer.step()
        if optimizer_norm is not None:
            optimizer_norm.step()

        batch_size = target.shape[0]
        n_examples += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["task_loss"] += float(task_loss.item()) * batch_size
        if aux_loss is not None:
            totals["aux_loss"] += float(aux_loss.item()) * batch_size

        if batch_idx % args.log_interval == 0:
            print(
                f"Train epoch {epoch} [{n_examples}/{len(train_loader.dataset)}] "
                f"loss={loss.item():.6f} task={task_loss.item():.6f} "
                f"aux={0.0 if aux_loss is None else aux_loss.item():.6f}"
            )
        if args.dry_run:
            break

    return {name: value / n_examples for name, value in totals.items()}


def evaluate(
    model: nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    *,
    aux_loss_weight: float = 1.0,
    dry_run: bool = False,
) -> dict[str, float | int]:
    model.eval()
    total_task_loss = 0.0
    total_aux_loss = 0.0
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            batch_size = target.shape[0]
            total_task_loss += float(criterion_sum(logits, target).item())
            aux_loss = getattr(model, "last_aux_loss", None)
            if aux_loss is not None:
                total_aux_loss += float(aux_loss.item()) * batch_size
            n_correct += int((logits.argmax(dim=1) == target).sum().item())
            n_total += batch_size
            if dry_run:
                break

    task_loss = total_task_loss / n_total
    aux_loss = total_aux_loss / n_total
    return {
        "loss": task_loss,
        "task_loss": task_loss,
        "aux_loss": aux_loss,
        "joint_loss": task_loss + aux_loss_weight * aux_loss,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy_pct": 100.0 * n_correct / n_total,
    }


def log_epoch(
    epoch: int,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float | int],
    *,
    use_wandb: bool,
) -> None:
    print(
        f"Test epoch {epoch}: loss={test_metrics['loss']:.6f}, "
        f"accuracy={test_metrics['n_correct']}/{test_metrics['n_total']} "
        f"({test_metrics['accuracy_pct']:.2f}%)"
    )
    if use_wandb:
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/task_loss": train_metrics["task_loss"],
                "train/aux_loss": train_metrics["aux_loss"],
                "test/loss": test_metrics["loss"],
                "test/accuracy_pct": test_metrics["accuracy_pct"],
                "validation/task_loss": test_metrics["task_loss"],
                "validation/aux_loss": test_metrics["aux_loss"],
                "validation/joint_loss": test_metrics["joint_loss"],
            }
        )
