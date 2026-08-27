"""Train an E-only or E/I dense network, optionally with hard-coded normalization.

Covers Figure 2 of the paper (LN vs no-LN, E/I vs E-only) and supplies the
"No-Norm" reference condition used in Figure 3.

Example::

    python -m experiments.dense_fmnist.train_ei_network \
        --data.brightness_factor=0.75 --model.layer_norm=1 \
        --model.excitation_training=0 --train.epochs=50
"""

import numpy as np
import torch

from experiments.dense_fmnist.cli import build_parser, wandb_config
from experiments.dense_fmnist.optim import build_optimizer
from experiments.dense_fmnist.training import resolve_device, run_training
from inhibition.data import make_dense_fashion_mnist_dataloaders
from inhibition.densenet import EDenseNet, EIDenseNet, norm_type_from_flags


def build_model(args, *, wandb_log: bool):
    norm_type = norm_type_from_flags(args.normtype, args.divisive_norm, args.layer_norm)
    # In this entry point excitation_training selects the architecture:
    # 1 ablates the inhibitory units, 0 keeps the full E/I network.
    net_cls = EDenseNet if args.excitation_training else EIDenseNet
    return net_cls(
        hidden_size=args.hidden_layer_width,
        output_size=args.n_outputs,
        num_layers=args.num_layers,
        norm_type=norm_type,
        normtype_detach=bool(args.normtype_detach),
        track_alignment=bool(args.track_alignment),
        wandb_log=wandb_log,
    )


def main():
    args = build_parser(homeostasis_defaults=False).parse_args()
    torch.manual_seed(args.seed)
    # inhibition.init draws from numpy's global RNG, so seed that too.
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    train_loader, test_loader = make_dense_fashion_mnist_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_accel=device.type != "cpu",
        brightness_factor=args.brightness_factor,
        brightness_factor_eval=args.brightness_factor_eval,
        dataset=args.dataset,
    )

    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            config=wandb_config(args),
        )

    model = build_model(args, wandb_log=bool(args.use_wandb)).to(device)
    optimizer = build_optimizer(model, args)

    try:
        run_training(args, model, device, train_loader, test_loader, optimizer)
    finally:
        if args.use_wandb:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
