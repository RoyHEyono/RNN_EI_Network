"""Optimizer construction with separate excitatory and inhibitory learning rates."""

import torch


#: Suffixes routed to the ``inhib_lrs.wix`` group (E -> I projections) and the
#: ``inhib_lrs.wei`` group (I -> E projections).
WIX_SUFFIXES = ("W_IE", "U_IE")
WEI_SUFFIXES = ("W_EI", "U_EI")


def param_groups(model, args):
    """Split parameters into excitatory / E->I / I->E groups.

    Inhibitory plasticity is treated as operating on its own timescale: those
    groups get their own learning rate, their own momentum, and no weight decay
    (matching the reference implementation).
    """
    exc, wix, wei = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(WIX_SUFFIXES):
            wix.append(p)
        elif name.endswith(WEI_SUFFIXES):
            wei.append(p)
        else:
            exc.append(p)

    lr_wix = args.lr_wix if args.use_sep_inhib_lrs else args.lr
    lr_wei = args.lr_wei if args.use_sep_inhib_lrs else args.lr

    groups = [
        {"params": exc, "lr": args.lr, "momentum": args.momentum, "weight_decay": args.wd},
        {"params": wix, "lr": lr_wix, "momentum": args.inhib_momentum, "weight_decay": 0.0},
        {"params": wei, "lr": lr_wei, "momentum": args.inhib_momentum, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def build_optimizer(model, args):
    if args.algorithm != "sgd":
        raise ValueError(f"unsupported optimizer {args.algorithm!r}")
    return torch.optim.SGD(param_groups(model, args))
