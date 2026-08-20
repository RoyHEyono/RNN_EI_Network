"""Build the paper's data figures from cached wandb run summaries.

    python -m experiments.dense_fmnist.figures.make_figures \
        --entity YOUR_ENTITY --project Luminosity_LNHomeostasis --all

Figure 1 is a schematic; only its stimulus panel (1b) is generated here, and it
needs no wandb access.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.dense_fmnist.figures import conditions as C
from experiments.dense_fmnist.figures import wandb_io as W
from experiments.dense_fmnist.figures.panels import (
    alignment_histograms,
    boxplot_with_significance,
    scatter_vs_diagonal,
    two_group_boxplot,
)
from experiments.dense_fmnist.figures.style import (
    COLOR_E_ONLY,
    COLOR_LN,
    EPSILONS,
    save,
    set_format,
    use_paper_style,
)

#: The paper reports the best 10 hyperparameter configurations per condition.
TOP_K = 10

GRADNORM_COMPONENTS = ("scale", "decorrelate", "center", "full")


def _acc(runs):
    return [r["summary"]["test_acc"] for r in runs if r["summary"].get("test_acc") is not None]


def _paired(runs, cond_x, cond_y, epsilons=EPSILONS):
    return {
        eps: W.pair_by_config(W.select(runs, **cond_x(eps)), W.select(runs, **cond_y(eps)))
        for eps in epsilons
    }


def _pooled_top(runs, cond, epsilons=EPSILONS, k=TOP_K):
    """Best ``k`` runs of a condition at each luminance, pooled together."""
    out = []
    for eps in epsilons:
        out.extend(W.top_k(W.select(runs, **cond(eps)), k))
    return out


# --------------------------------------------------------------------------
# Figure 2: hard-coded layer normalization improves perceptual invariance
# --------------------------------------------------------------------------

def figure2(runs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_vs_diagonal(
        ax,
        _paired(runs, C.ei_ln, C.ei_no_ln),
        xlabel="LN (Acc %)",
        ylabel="No LN (Acc %)",
    )
    save(fig, out_dir, "figure2a")

    ei = _acc(_pooled_top(runs, C.ei_ln))
    e_only = _acc(_pooled_top(runs, C.e_only_ln))
    fig, ax = plt.subplots(figsize=(5, 5))
    p = two_group_boxplot(ax, ei, e_only, "E-I", "E-only",
                          colors=[COLOR_LN, COLOR_E_ONLY], ylabel="Acc % (Top 10)")
    ax.set_xlabel("LN")
    print(f"  figure2b Mann-Whitney p = {p:.4g}")
    save(fig, out_dir, "figure2b")


# --------------------------------------------------------------------------
# Figure 3: learned inhibition normalizes excitatory activity
# --------------------------------------------------------------------------

def figure3(runs, out_dir):
    groups = {
        "No-Norm": C.ei_no_ln,
        "I-Norm (sub)": C.inorm_subtractive,
        "I-Norm": C.inorm,
    }
    moments = {"mu": {}, "var": {}}
    for label, cond in groups.items():
        selected = _pooled_top(runs, cond)
        moments["mu"][label] = [
            v for v in (W.layer_mean(r, "train_", ("fc0_mu", "fc1_mu")) for r in selected)
            if v is not None
        ]
        moments["var"][label] = [
            v for v in (W.layer_mean(r, "train_", ("fc0_var", "fc1_var")) for r in selected)
            if v is not None
        ]

    fig, axes = plt.subplots(2, 1, figsize=(6, 7))
    face = ["lightgray", "darkgray", "white"]
    for ax, key, target, ylabel in (
        (axes[0], "mu", 0.0, "First Moment"),
        (axes[1], "var", 1.0, "Second Moment"),
    ):
        data = [moments[key][label] for label in groups]
        box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(list(groups))
        for patch, color in zip(box["boxes"], face):
            patch.set(facecolor=color, linewidth=2.5)
        ax.axhline(target, color="black", linestyle=":", lw=1.5)
        ax.set_ylabel(ylabel)
        if key == "var":
            ax.set_yscale("log")
    fig.tight_layout()
    save(fig, out_dir, "figure3b")


# --------------------------------------------------------------------------
# Figure 4: I-Norm normalizes activity but does not recover LN's learning gain
# --------------------------------------------------------------------------

def _alignment_values(runs, cond, eps, prefix):
    return [
        v for v in (W.layer_mean(r, prefix) for r in W.select(runs, **cond(eps)))
        if v is not None
    ]


def figure4(runs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_vs_diagonal(
        ax,
        _paired(runs, C.ei_ln, C.inorm),
        xlabel="LN (Acc %)",
        ylabel="I-Norm (Acc %)",
    )
    save(fig, out_dir, "figure4a")

    eps = 0.75
    fig, _ = alignment_histograms(
        {
            "I-Norm vs LN Output Alignment": _alignment_values(runs, C.inorm, eps, "output_alignment_"),
            "I-Norm vs LN Gradient Alignment": _alignment_values(runs, C.inorm, eps, "gradient_alignment_"),
        },
        xlabel=rf"Cosine similarity to LN ($\epsilon$={eps})",
        bins=10,
        xlim=(0, 1),
    )
    save(fig, out_dir, "figure4b")


# --------------------------------------------------------------------------
# Figure 5: hard-coded LN gradients in I-Norm networks restore LN performance
# --------------------------------------------------------------------------

def figure5(runs, out_dir):
    eps = 0.75
    cond = lambda e: C.inorm_gradnorm(e, "full")

    fig, axes = alignment_histograms(
        {
            "I-Norm vs LN Output Alignment": _alignment_values(runs, cond, eps, "output_alignment_"),
            "I-Norm vs LN Gradient Alignment": _alignment_values(runs, cond, eps, "gradient_alignment_"),
        },
        xlabel=rf"Cosine similarity to LN ($\epsilon$={eps})",
        bins=20,
        xlim=(0.96, 1.0),
    )
    save(fig, out_dir, "figure5b")

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_vs_diagonal(
        ax,
        _paired(runs, C.ei_ln, cond),
        xlabel="LN (Acc %)",
        ylabel="I-Norm w/ GradNorm (Acc %)",
    )
    save(fig, out_dir, "figure5c")


# --------------------------------------------------------------------------
# Figure 6: which component of the LN gradient matters
# --------------------------------------------------------------------------

def figure6(runs, out_dir, epsilons=(0.0, 0.75)):
    fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eps in zip(axes, epsilons):
        groups = [
            _acc(W.select(runs, **C.inorm_gradnorm(eps, fb))) for fb in GRADNORM_COMPONENTS
        ]
        boxplot_with_significance(
            ax,
            groups,
            [fb.capitalize() for fb in GRADNORM_COMPONENTS],
            baseline=_acc(W.select(runs, **C.ei_ln(eps))),
            ylabel="Test Acc" if eps == epsilons[0] else "",
        )
        ax.set_title(rf"$\epsilon$={eps}")
    fig.tight_layout()
    save(fig, out_dir, "figure6a")

    hist_eps = 0.75
    fig, _ = alignment_histograms(
        {
            fb.capitalize(): _alignment_values(
                runs, lambda e, f=fb: C.inorm_gradnorm(e, f), hist_eps, "gradient_alignment_"
            )
            for fb in ("center", "decorrelate", "scale")
        },
        xlabel="I-Norm vs LN Gradient Alignment",
        bins=20,
        xlim=(0, 0.65),
        figsize=(6, 6),
    )
    save(fig, out_dir, "figure6b")


# --------------------------------------------------------------------------
# Figure 7: gradient centering by fixed random lateral inhibition
# --------------------------------------------------------------------------

def figure7(runs, out_dir, epsilons=(0.0, 0.75)):
    variants = (
        ("lat. inhib.", C.inorm_lateral_inhibition),
        ("center", lambda e: C.inorm_gradnorm(e, "center")),
        ("full", lambda e: C.inorm_gradnorm(e, "full")),
    )
    fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eps in zip(axes, epsilons):
        groups = [_acc(W.select(runs, **cond(eps))) for _, cond in variants]
        boxplot_with_significance(
            ax,
            groups,
            [label for label, _ in variants],
            baseline=_acc(W.select(runs, **C.ei_ln(eps))),
            ylabel="Test Acc" if eps == epsilons[0] else "",
        )
        ax.set_title(rf"$\epsilon$={eps}")
    fig.tight_layout()
    save(fig, out_dir, "figure7b")


FIGURES = {2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6, 7: figure7}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", required=True, help="wandb entity")
    p.add_argument("--project", default="Luminosity_LNHomeostasis")
    p.add_argument("--out", type=Path, default=Path("figures_out"))
    p.add_argument("--refresh", action="store_true", help="re-download run summaries")
    p.add_argument("--figures", type=int, nargs="+", choices=sorted(FIGURES))
    p.add_argument("--all", action="store_true")
    p.add_argument("--format", default="svg", choices=["svg", "pdf", "png"])
    args = p.parse_args()

    if not args.all and not args.figures:
        p.error("pass --all or --figures N [N ...]")

    use_paper_style()
    set_format(args.format)
    runs = W.load_runs(args.entity, args.project, refresh=args.refresh)
    print(f"{len(runs)} runs available")

    for number in sorted(FIGURES) if args.all else args.figures:
        print(f"figure {number}:")
        FIGURES[number](runs, args.out)


if __name__ == "__main__":
    main()
