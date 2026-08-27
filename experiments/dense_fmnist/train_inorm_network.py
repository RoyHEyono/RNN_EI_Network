"""Train an I-Norm network: inhibition learns to layer-normalize excitatory activity.

Covers Figures 3-7. The condition is selected by three flags:

* ``--model.shunting``       0 = subtractive-only I-Norm, 1 = subtractive + divisive
* ``--model.normtype_detach`` 1 = plain I-Norm (Fig. 4), 0 = I-Norm + GradNorm (Fig. 5)
* ``--model.ln_feedback``    which part of the LayerNorm Jacobian GradNorm imposes
  (``full``/``center``/``scale``/``decorrelate`` for Fig. 6, ``fa_center`` --
  the fixed random lateral-inhibition pool -- for Fig. 7)

Example::

    python -m experiments.dense_fmnist.train_inorm_network \
        --data.brightness_factor=0.75 --model.normtype_detach=0 \
        --model.ln_feedback=center --opt.lambda_homeo_var=0.01
"""

import numpy as np
import torch

from experiments.dense_fmnist.cli import build_parser, wandb_config
from experiments.dense_fmnist.optim import build_optimizer
from experiments.dense_fmnist.training import resolve_device, run_training
from inhibition.data import make_dense_fashion_mnist_dataloaders
from inhibition.densenet import INormDenseNet


def build_model(args, *, wandb_log: bool):
    return INormDenseNet(
        hidden_size=args.hidden_layer_width,
        output_size=args.n_outputs,
        num_layers=args.num_layers,
        ln_feedback=args.ln_feedback,
        # normtype_detach detaches the normalization from the backward pass,
        # i.e. it turns GradNorm off.
        gradient_norm=not bool(args.normtype_detach),
        shunting=bool(args.shunting),
        track_alignment=bool(args.track_alignment),
        freeze_ei=bool(args.freeze_ei),
        wandb_log=wandb_log,
    )


def main():
    args = build_parser(homeostasis_defaults=True).parse_args()
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
