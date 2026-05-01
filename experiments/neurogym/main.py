from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from experiments.neurogym.cli import build_train_arg_parser
from experiments.neurogym.training import (
    configure_neurogym_warnings,
    require_neurogym,
    train_supervised_steps,
    trial_eval_accuracy,
)
from inhibition.model import NeurogymRNNNet, NeurogymVanillaRNNNet, inorm_param_groups


def main() -> None:
    require_neurogym()
    configure_neurogym_warnings()
    import neurogym as ngym

    args = build_train_arg_parser().parse_args()
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
        ).to(device)
    else:
        model = NeurogymVanillaRNNNet(
            ob_size=ob_size,
            hidden_size=args.hidden,
            n_actions=act_size,
            nonlinearity=args.nonlinearity,
            num_layers=args.rnn_layers,
            ffn_hidden=args.ffn_hidden,
            use_layer_norm=not args.no_vanilla_layer_norm,
        ).to(device)

    criterion = nn.CrossEntropyLoss()

    lr_ie = args.lr if args.lr_ie is None else args.lr_ie
    lr_ei = args.lr if args.lr_ei is None else args.lr_ei
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
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

    train_supervised_steps(args, model, dataset, criterion, optimizer, device)
    print("Finished training.")

    if args.eval_trials > 0:
        acc = trial_eval_accuracy(model, env, device, args.eval_trials)
        print(f"Trial eval ({args.eval_trials} trials): mean accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
