"""
Train :class:`~inhibition.model.NeurogymRNNNet` or the vanilla control
:class:`~inhibition.model.NeurogymVanillaRNNNet` on a NeuroGym supervised dataset.

Follows the flow from the NeuroGym PyTorch supervised example (Dataset batching,
CrossEntropy over flattened time × batch, optional trial-wise evaluation):
https://neurogym.github.io/neurogym/latest/examples/supervised_learning_pytorch/
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from inhibition.model import NeurogymRNNNet, NeurogymVanillaRNNNet, inorm_param_groups


def _require_neurogym() -> None:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroGym supervised RNN (EI vs vanilla control)")
    p.add_argument(
        "--arch",
        type=str,
        default="vanilla",
        choices=("ei", "vanilla"),
        help="ei: SimpleEERNN+EiDenseLayer; vanilla: nn.RNN+MLP head (control)",
    )
    p.add_argument(
        "--task",
        type=str,
        default="PerceptualDecisionMaking-v0",
        help="NeuroGym task id (default: PerceptualDecisionMaking-v0)",
    )
    p.add_argument("--dt", type=int, default=100, help="env dt (default: 100)")
    p.add_argument("--seq-len", type=int, default=100, metavar="T", help="dataset seq_len")
    p.add_argument("--batch-size", type=int, default=16, metavar="N")
    p.add_argument("--epochs", type=int, default=2000, help="training steps (each step = one dataset batch)")
    p.add_argument("--hidden", type=int, default=64, metavar="H", help="RNN hidden size")
    p.add_argument("--nonlinearity", type=str, default="relu", choices=("tanh", "relu"))
    p.add_argument(
        "--rnn-layers",
        type=int,
        default=1,
        metavar="L",
        help="stacked RNN depth (vanilla --arch only; default 1)",
    )
    p.add_argument(
        "--ffn-hidden",
        type=int,
        default=None,
        metavar="F",
        help="MLP hidden width in readout head (vanilla only; default = --hidden)",
    )
    p.add_argument(
        "--no-vanilla-layer-norm",
        action="store_true",
        help="disable LayerNorm on RNN outputs (vanilla --arch only; default: LN on, like SimpleEERNN)",
    )
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-trials", type=int, default=200, help="trial-based eval count (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="cuda | cpu (empty = auto)")
    p.add_argument("--optimizer", type=str, default="sgd", choices=("adam", "sgd"))
    p.add_argument("--lr", type=float, default=1e-2, help="Adam lr or SGD base lr for excitatory group")
    p.add_argument("--lr-ie", type=float, default=None, dest="lr_ie", help="SGD lr for *_IE (default: --lr)")
    p.add_argument("--lr-ei", type=float, default=None, dest="lr_ei", help="SGD lr for *_EI (default: --lr)")
    p.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (default 0)")
    return p.parse_args()


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


def trial_eval_accuracy(model: nn.Module, env, device: torch.device, num_trial: int) -> float:
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


def main() -> None:
    _require_neurogym()
    import neurogym as ngym

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

    args = parse_args()
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

        running_loss += float(loss.item())
        if (i + 1) % args.log_interval == 0:
            print(f"step {i + 1}  mean_loss_last_{args.log_interval}: {running_loss / args.log_interval:.5f}")
            running_loss = 0.0

    print("Finished training.")

    if args.eval_trials > 0:
        acc = trial_eval_accuracy(model, env, device, args.eval_trials)
        print(f"Trial eval ({args.eval_trials} trials): mean accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
