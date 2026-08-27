"""Command-line flags for the dense Fashion-MNIST I-Norm experiments.

Flag names deliberately mirror the original ``HomeostaticDANN`` fastargs
sections (``--model.normtype``, ``--opt.inhib_lrs.wei``, ...) so the paper's
batch scripts port over unchanged, and :func:`wandb_config` emits the same
*flat* config keys the figure notebooks filter on.
"""

import argparse


def build_parser(*, homeostasis_defaults: bool) -> argparse.ArgumentParser:
    """Build the shared parser.

    ``homeostasis_defaults`` picks the defaults of the I-Norm entry point
    (``homeostasis=1``, ``shunting=1``) over the plain E/EI one.
    """
    p = argparse.ArgumentParser(
        description="Dense Fashion-MNIST training for the I-Norm paper figures"
    )

    train = p.add_argument_group("train")
    train.add_argument(
        "--train.dataset",
        dest="dataset",
        type=str,
        default="fashionmnist",
        choices=["fashionmnist", "fashionmnist_contrast"],
        help="fashionmnist: luminance jitter; fashionmnist_contrast: contrast jitter "
        "(same --data.brightness_factor epsilon)",
    )
    train.add_argument("--train.batch_size", dest="batch_size", type=int, default=32)
    train.add_argument("--train.test_batch_size", dest="test_batch_size", type=int, default=512)
    train.add_argument("--train.epochs", dest="epochs", type=int, default=50)
    train.add_argument("--train.seed", dest="seed", type=int, default=0)
    train.add_argument("--train.use_testset", dest="use_testset", type=int, default=1)

    data = p.add_argument_group("data")
    data.add_argument("--data.brightness_factor", dest="brightness_factor", type=float, default=0.75,
                      help="epsilon of the train-time jitter: luminance Δ~U(-eps,+eps) for "
                      "fashionmnist, or contrast scale c~U(1-eps,1+eps) for fashionmnist_contrast")
    data.add_argument("--data.brightness_factor_eval", dest="brightness_factor_eval", type=float, default=0.0,
                      help="fixed test-time jitter amount; 0 means jitter the test set like training")
    data.add_argument("--data.data_dir", dest="data_dir", type=str, default="./data")

    model = p.add_argument_group("model")
    model.add_argument("--model.normtype", dest="normtype", type=int, default=0,
                       help="1 = mean-normalization only")
    model.add_argument("--model.divisive_norm", dest="divisive_norm", type=int, default=0,
                       help="1 = divisive normalization only")
    model.add_argument("--model.layer_norm", dest="layer_norm", type=int, default=0,
                       help="1 = hard-coded LayerNorm")
    model.add_argument("--model.normtype_detach", dest="normtype_detach", type=int, default=0,
                       help="1 detaches the normalization from the backward pass (LN- / no GradNorm)")
    model.add_argument("--model.excitation_training", dest="excitation_training", type=int, default=1,
                       help="in the EI entry point, 1 selects the excitatory-only network")
    model.add_argument("--model.shunting", dest="shunting", type=int,
                       default=1 if homeostasis_defaults else 0,
                       help="1 enables the divisive inhibitory pathway of the I-Norm layer")
    model.add_argument("--model.ln_feedback", dest="ln_feedback", type=str, default="full",
                       choices=["full", "center", "scale", "decorrelate", "fa_center"])
    model.add_argument("--model.hidden_layer_width", dest="hidden_layer_width", type=int, default=234)
    model.add_argument("--model.num_layers", dest="num_layers", type=int, default=1,
                       help="num_layers=1 gives fc0+fc1, i.e. two hidden layers")
    model.add_argument("--model.freeze_ei", dest="freeze_ei", type=int, default=1,
                       help="keep W_EI/U_EI at their initialized averaging values")
    model.add_argument("--model.track_alignment", dest="track_alignment", type=int, default=1,
                       help="log cosine similarity to LayerNorm; roughly doubles step time")
    # Recorded for wandb-query compatibility with the original figure notebooks;
    # these do not change behaviour, they label which condition a run belongs to.
    model.add_argument("--model.homeostasis", dest="homeostasis", type=int,
                       default=1 if homeostasis_defaults else 0)
    model.add_argument("--model.feedback_alignment", dest="feedback_alignment", type=int, default=0)
    model.add_argument("--model.task_opt_inhib", dest="task_opt_inhib", type=int, default=0)
    model.add_argument("--model.homeo_opt_exc", dest="homeo_opt_exc", type=int, default=0)
    model.add_argument("--model.is_dann", dest="is_dann", type=int, default=1)
    model.add_argument("--model.n_outputs", dest="n_outputs", type=int, default=10)

    opt = p.add_argument_group("opt")
    opt.add_argument("--opt.algorithm", dest="algorithm", type=str, default="sgd", choices=["sgd"])
    opt.add_argument("--opt.lr", dest="lr", type=float, default=0.0098)
    opt.add_argument("--opt.wd", dest="wd", type=float, default=0.0)
    opt.add_argument("--opt.momentum", dest="momentum", type=float, default=0.0)
    opt.add_argument("--opt.inhib_momentum", dest="inhib_momentum", type=float, default=0.0)
    opt.add_argument("--opt.use_sep_inhib_lrs", dest="use_sep_inhib_lrs", type=int, default=1)
    opt.add_argument("--opt.inhib_lrs.wei", dest="lr_wei", type=float, default=0.00294)
    opt.add_argument("--opt.inhib_lrs.wix", dest="lr_wix", type=float, default=0.64646)
    opt.add_argument("--opt.lambda_homeo", dest="lambda_homeo", type=float, default=0.01)
    opt.add_argument("--opt.lambda_homeo_var", dest="lambda_homeo_var", type=float, default=0.01)

    exp = p.add_argument_group("exp")
    exp.add_argument("--exp.use_wandb", dest="use_wandb", type=int, default=0)
    exp.add_argument("--exp.wandb_project", dest="wandb_project", type=str, default="Luminosity_LNHomeostasis")
    exp.add_argument("--exp.wandb_entity", dest="wandb_entity", type=str, default="")
    exp.add_argument("--exp.log_interval", dest="log_interval", type=int, default=100)
    exp.add_argument("--exp.log_grad_norms", dest="log_grad_norms", type=int, default=1)
    exp.add_argument("--exp.device", dest="device", type=str, default="auto")
    exp.add_argument("--exp.dry_run", dest="dry_run", action="store_true",
                     help="stop each epoch after a handful of batches")
    return p


# Keys the figure notebooks read from run.config. They are logged flat (no
# section prefix), matching the reference repo's get_params_to_log_wandb.
WANDB_CONFIG_KEYS = (
    "dataset", "batch_size", "test_batch_size", "epochs", "seed", "use_testset",
    "brightness_factor", "brightness_factor_eval",
    "normtype", "divisive_norm", "layer_norm", "normtype_detach",
    "excitation_training", "shunting", "ln_feedback", "hidden_layer_width",
    "num_layers", "freeze_ei", "track_alignment",
    "homeostasis", "feedback_alignment", "task_opt_inhib", "homeo_opt_exc",
    "is_dann", "n_outputs",
    "algorithm", "lr", "wd", "momentum", "inhib_momentum", "use_sep_inhib_lrs",
    "lambda_homeo", "lambda_homeo_var",
)


def wandb_config(args) -> dict:
    """Flat config dict for wandb, including the nested ``inhib_lrs`` mapping.

    The notebooks pair hyperparameter-matched runs by comparing
    ``['lr', 'wd', 'inhib_lrs', 'momentum', 'inhib_momentum']``, so
    ``inhib_lrs`` must be a dict rather than two separate scalars.
    """
    cfg = {k: getattr(args, k) for k in WANDB_CONFIG_KEYS}
    # Logged as a real bool so `{"config.use_testset": True}` queries match.
    cfg["use_testset"] = bool(args.use_testset)
    cfg["inhib_lrs"] = {"wei": args.lr_wei, "wix": args.lr_wix}
    return cfg
