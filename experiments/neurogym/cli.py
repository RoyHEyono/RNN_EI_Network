import argparse


def build_train_arg_parser() -> argparse.ArgumentParser:
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
    p.add_argument(
        "--eval-trials",
        type=int,
        default=50,
        help="trial-rollout eval after each supervised step (0 to skip; expensive)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="cuda | cpu (empty = auto)")
    p.add_argument("--optimizer", type=str, default="sgd", choices=("adam", "sgd"))
    p.add_argument("--lr", type=float, default=1e-2, help="Adam lr or SGD base lr for excitatory group")
    p.add_argument("--lr-ie", type=float, default=None, dest="lr_ie", help="SGD lr for *_IE (default: --lr)")
    p.add_argument("--lr-ei", type=float, default=None, dest="lr_ei", help="SGD lr for *_EI (default: --lr)")
    p.add_argument("--momentum", type=float, default=0.0, help="SGD momentum (default 0)")
    p.add_argument(
        "--wandb",
        action="store_true",
        help="log metrics to Weights & Biases",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="neurogym-ei-rnn",
        metavar="NAME",
        help="W&B project when --wandb is set (default: neurogym-ei-rnn)",
    )
    return p
