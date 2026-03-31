import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR

from experiments.mnist.cli import build_train_arg_parser
from inhibition.data import make_mnist_dataloaders
from inhibition.model import Net, inorm_param_groups
from experiments.mnist.training import evaluate, format_eval_metrics, train_one_epoch


def main():
    parser = build_train_arg_parser()
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_loader, test_loader = make_mnist_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_accel=use_accel,
    )

    model = Net().to(device)
    optimizer = optim.SGD(
        inorm_param_groups(model, args.lr, args.lr_ie, args.lr_ei),
    )

    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model, log="all", log_freq=10)

    try:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_task_loss, train_local_ln = train_one_epoch(
                args, model, device, train_loader, optimizer, epoch
            )
            test_metrics = evaluate(model, device, test_loader)
            print(format_eval_metrics(test_metrics))
            if args.wandb:
                wandb.log({"train/loss": train_loss, "train/task_loss": train_task_loss, "train/local_ln_sum": train_local_ln, "test/loss": test_metrics["loss"], "test/accuracy_pct": test_metrics["accuracy_pct"], "epoch": epoch}, step=epoch)
            scheduler.step()
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
