import torch
import torch.optim as optim
import wandb

from experiments.cli import build_train_arg_parser
from inhibition.data import make_mnist_dataloaders, make_fashion_mnist_dataloaders
from inhibition.model import RNNNet, inorm_param_groups, DeepNet 
from experiments.training import evaluate, format_eval_metrics, train_one_epoch


def main():
    parser = build_train_arg_parser()
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_loader, test_loader = make_fashion_mnist_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_accel=use_accel,
        brightness_factor=args.brightness_factor,
    )
    # model = DeepNet().to(device)
    model = RNNNet().to(device)
    optimizer = optim.SGD(
        inorm_param_groups(model, args.lr, args.lr_ie, args.lr_ei),
    )

    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model, log="all", log_freq=100)

    try:
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(args, model, device, train_loader, optimizer, epoch)
            test_metrics = evaluate(model, device, test_loader)
            print(format_eval_metrics(test_metrics))
            if args.wandb:
                wandb.log({"test/loss": test_metrics["loss"], "test/accuracy_pct": test_metrics["accuracy_pct"], "epoch": epoch})
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
