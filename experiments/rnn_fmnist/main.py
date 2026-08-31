from __future__ import annotations

import os

import numpy as np
import torch
import torch.optim as optim
import wandb

from experiments.rnn_fmnist.cli import build_train_arg_parser
from experiments.rnn_fmnist.training import evaluate, log_epoch, train_one_epoch
from inhibition.data import make_dense_fashion_mnist_dataloaders
from inhibition.model import RNNNet, inorm_param_groups, param_ln_param_groups


def _device_from_args(args) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if not args.no_accel and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = build_train_arg_parser().parse_args()
    if args.epsilon < 0:
        raise SystemExit("--epsilon must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device_from_args(args)

    train_loader, test_loader = make_dense_fashion_mnist_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_accel=device.type != "cpu",
        brightness_factor=args.epsilon,
        brightness_factor_eval=args.eval_epsilon,
        download=not args.no_download,
        num_workers=args.num_workers,
        dataset=args.dataset,
    )

    use_param_ln = args.normalization == "paramln"
    model = RNNNet(
        hidden_size=args.hidden_size,
        nonlinearity=args.nonlinearity,
        use_parametrized_layer_norm=use_param_ln,
    ).to(device)
    optimizer = optim.SGD(
        inorm_param_groups(model, args.lr, args.lr_ie, args.lr_ei),
        momentum=args.momentum,
    )
    optimizer_norm = None
    if use_param_ln:
        optimizer_norm = optim.SGD(
            param_ln_param_groups(
                model.rnn.layer_norm,
                args.lr_norm_mean,
                args.lr_norm_var0,
                args.lr_norm_var2,
            ),
            momentum=args.momentum,
        )

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=os.environ.get("WANDB_RUN_NAME") or None,
        )
        wandb.watch(model, log="all", log_freq=max(args.log_interval, 50))

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                optimizer_norm=optimizer_norm,
            )
            test_metrics = evaluate(
                model,
                device,
                test_loader,
                aux_loss_weight=args.aux_loss_weight,
                dry_run=args.dry_run,
            )
            log_epoch(
                epoch, train_metrics, test_metrics, use_wandb=args.wandb
            )
            if args.dry_run:
                break
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
