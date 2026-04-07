import argparse


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=32, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=0.00547998500676349, metavar="LR", help="Adadelta lr for excitatory params (W_EE, bias) (default: 1.0)")
    parser.add_argument("--lr-ie", type=float, default=0.041498118034054156, metavar="LR", dest="lr_ie", help="Adadelta lr for *_IE weights (W_IE, U_IE) (default: 1.0)")
    parser.add_argument("--lr-ei", type=float, default=0.005825485335208354, metavar="LR", dest="lr_ei", help="Adadelta lr for *_EI weights (W_EI, U_EI) (default: 1.0)")
    parser.add_argument("--local-loss-weight", type=float, default=0.01, metavar="W", dest="local_loss_weight", help="weight for INormLayer local (moments) loss vs task NLL (default: 1.0)")
    parser.add_argument("--no-accel", action="store_true", help="disables accelerator")
    parser.add_argument("--dry-run", action="store_true", help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", help="For Saving the current Model")
    parser.add_argument("--data-dir", type=str, default="../data", help="directory for MNIST files (default: ../data)")
    parser.add_argument("--wandb", action="store_false", help="enable Weights & Biases (wandb.watch on the model)")
    parser.add_argument("--wandb-project", type=str, default="mnist-ei", metavar="NAME", help="W&B project name when --wandb is set (default: mnist-ei)")
    return parser
