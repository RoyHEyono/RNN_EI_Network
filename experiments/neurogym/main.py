from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from experiments.neurogym.cli import build_train_arg_parser
from experiments.neurogym.training import (
    configure_neurogym_warnings,
    pretrain_parametrized_layer_norm,
    require_neurogym,
    train_supervised_steps,
)
from inhibition.model import (
    NeurogymRNNNet,
    NeurogymVanillaLSTMNet,
    NeurogymVanillaRNNNet,
    inorm_param_groups,
)


def main() -> None:
    require_neurogym()
    configure_neurogym_warnings()
    import neurogym as ngym

    args = build_train_arg_parser().parse_args()
    if args.param_layer_norm and args.arch != "ei":
        raise SystemExit("--param-layer-norm requires --arch ei")
    torch.manual_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ngym.Dataset(
        args.task,
        env_kwargs={"dt": args.dt},
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    env = dataset.env
    ob_size = int(env.observation_space.shape[0])
    act_size = int(env.action_space.n)

    if args.arch == "ei":
        model = NeurogymRNNNet(
            ob_size=ob_size,
            hidden_size=args.hidden,
            n_actions=act_size,
            nonlinearity=args.nonlinearity,
            use_parametrized_layer_norm=args.param_layer_norm,
        ).to(device)
    elif args.arch == "lstm":
        model = NeurogymVanillaLSTMNet(
            ob_size=ob_size,
            hidden_size=args.hidden,
            n_actions=act_size,
            nonlinearity=args.nonlinearity,
            num_layers=args.rnn_layers,
            use_layer_norm=args.layer_norm,
        ).to(device)
    else:
        model = NeurogymVanillaRNNNet(
            ob_size=ob_size,
            hidden_size=args.hidden,
            n_actions=act_size,
            nonlinearity=args.nonlinearity,
            num_layers=args.rnn_layers,
            use_layer_norm=args.layer_norm,
        ).to(device)

    criterion = nn.CrossEntropyLoss()

    norm_params = (
        list(model.rnn.layer_norm.parameters())
        if (args.arch == "ei" and args.param_layer_norm)
        else []
    )

    lr_ie = args.lr if args.lr_ie is None else args.lr_ie
    lr_ei = args.lr if args.lr_ei is None else args.lr_ei
    if args.optimizer == "adam":
        if norm_params:
            norm_param_ids = {id(p) for p in norm_params}
            main_params = [p for p in model.parameters() if id(p) not in norm_param_ids]
        else:
            main_params = list(model.parameters())
        optimizer = optim.Adam(main_params, lr=args.lr)
    else:
        if args.arch == "ei":
            optimizer = optim.SGD(
                inorm_param_groups(model, args.lr, lr_ie, lr_ei),
                momentum=args.momentum,
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
            )

    optimizer_norm = optim.Adam(norm_params, lr=args.lr_norm) if norm_params else None

    if args.wandb:
        run_name = os.environ.get("WANDB_RUN_NAME")
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=run_name or None,
        )
        wandb.watch(model, log="all", log_freq=max(args.log_interval, 50))

    try:
        if optimizer_norm is not None and args.param_ln_pretrain_steps > 0:
            pretrain_parametrized_layer_norm(
                model,
                dataset,
                optimizer_norm,
                args.param_ln_pretrain_steps,
                device,
                log_interval=args.log_interval,
                use_wandb=args.wandb,
            )
        train_supervised_steps(
            args,
            model,
            dataset,
            env,
            criterion,
            optimizer,
            device,
            optimizer_norm=optimizer_norm,
            aux_loss_weight=args.aux_loss_weight,
        )
        print("Finished training.")
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
