import argparse


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EI-readout RNN on brightness/contrast Fashion-MNIST"
    )
    parser.add_argument(
        "--normalization",
        choices=("ln", "paramln"),
        default="ln",
        help="normalization inside the recurrent update",
    )
    parser.add_argument(
        "--dataset",
        choices=("fashionmnist", "fashionmnist_contrast"),
        default="fashionmnist",
        help="brightness or contrast jitter, respectively",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="training jitter magnitude on raw [0,1] pixels",
    )
    parser.add_argument(
        "--eval-epsilon",
        "--eval_epsilon",
        type=float,
        default=0.0,
        dest="eval_epsilon",
        help="fixed evaluation jitter; zero reuses random training jitter",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--nonlinearity", choices=("tanh", "relu"), default="relu"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="", help="cpu | cuda (empty = auto)")
    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr-ie", "--lr_ie", type=float, default=0.001, dest="lr_ie"
    )
    parser.add_argument(
        "--lr-ei", "--lr_ei", type=float, default=0.1, dest="lr_ei"
    )
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument(
        "--lr-norm-mean",
        "--lr_norm_mean",
        type=float,
        default=1e-5,
        dest="lr_norm_mean",
    )
    parser.add_argument(
        "--lr-norm-var0",
        "--lr_norm_var0",
        type=float,
        default=1e-6,
        dest="lr_norm_var0",
    )
    parser.add_argument("--aux-loss-weight", type=float, default=1.0)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        type=str,
        default="rnn-fmnist-invariance",
        dest="wandb_project",
    )
    return parser
