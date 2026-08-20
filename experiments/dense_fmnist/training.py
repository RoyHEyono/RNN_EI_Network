"""Training and evaluation loop shared by both dense entry points.

The wandb key names here are load-bearing: the figure notebooks query
``test_acc``, ``train_fc{i}_mu`` / ``_var``, ``output_alignment_fc{i}`` and
``gradient_alignment_fc{i}`` by exact name. Per-batch metrics are logged with
``commit=False`` and flushed once per epoch by the ``commit=True`` call in
:func:`log_epoch`, which is what pairs them to an epoch step in the history.
"""

from __future__ import annotations

import torch
from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _grad_norms(model) -> dict:
    return {
        f"grad_norm/{name}": p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def train_epoch(args, model, device, loader, optimizer, *, use_wandb: bool):
    """One pass over the training set. Returns ``(mean_loss, accuracy_pct)``."""
    model.train()
    model.set_eval_logging(False)
    wandb = _maybe_wandb(use_wandb)

    total_loss, n_correct, n_total = 0.0, 0, 0
    lambda_homeo_var = getattr(args, "lambda_homeo_var", 0.0)
    wants_local_loss = hasattr(model, "local_loss")

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        task_loss = criterion(output, target)
        loss = task_loss
        local_diagnostics = {}
        if wants_local_loss:
            moments, local_diagnostics = model.local_loss()
            loss = task_loss + lambda_homeo_var * moments

        loss.backward()

        batch_acc = output.argmax(dim=1).eq(target).float().mean().item()
        if wandb is not None:
            payload = {
                "update_acc": batch_acc,
                "update_loss": task_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            for name, value in local_diagnostics.items():
                payload[f"train_{name}_local_loss"] = value
            if args.log_grad_norms:
                payload.update(_grad_norms(model))
            wandb.log(payload, commit=False)

        optimizer.step()

        total_loss += task_loss.item() * target.shape[0]
        n_correct += output.argmax(dim=1).eq(target).sum().item()
        n_total += target.shape[0]

        if args.dry_run and batch_idx >= 4:
            break

    return total_loss / n_total, 100.0 * n_correct / n_total


@torch.no_grad()
def eval_model(args, model, device, loader, *, use_wandb: bool):
    """Evaluate on the test set. Returns ``(mean_loss, accuracy_pct)``."""
    model.eval()
    model.set_eval_logging(True)
    total_loss, n_correct, n_total = 0.0, 0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item() * target.shape[0]
        n_correct += output.argmax(dim=1).eq(target).sum().item()
        n_total += target.shape[0]
        if args.dry_run and batch_idx >= 4:
            break
    model.set_eval_logging(False)
    return total_loss / n_total, 100.0 * n_correct / n_total


def log_epoch(use_wandb: bool, epoch, train_loss, train_acc, test_loss, test_acc):
    """The one committed log per epoch; it flushes the accumulated batch metrics."""
    wandb = _maybe_wandb(use_wandb)
    if wandb is None:
        return
    wandb.log(
        {
            "epoch_i": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )


def _maybe_wandb(use_wandb: bool):
    if not use_wandb:
        return None
    import wandb

    return wandb


def run_training(args, model, device, train_loader, test_loader, optimizer):
    """Full training run, returning the per-epoch results dict."""
    use_wandb = bool(args.use_wandb)
    results = {"test_losses": [], "test_accs": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            args, model, device, train_loader, optimizer, use_wandb=use_wandb
        )
        test_loss, test_acc = eval_model(
            args, model, device, test_loader, use_wandb=use_wandb
        )
        log_epoch(use_wandb, epoch, train_loss, train_acc, test_loss, test_acc)
        results["test_losses"].append(test_loss)
        results["test_accs"].append(test_acc)
        if epoch % max(1, args.log_interval // 100) == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:5.2f}% "
                f"| test loss {test_loss:.4f} acc {test_acc:5.2f}%",
                flush=True,
            )

    if use_wandb:
        import wandb

        wandb.summary["test_loss_auc"] = float(sum(results["test_losses"]))
    return results
